#include <Arduino.h>
#include <driver/i2s.h>
#include <esp_timer.h>

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "tinyml_model_params.h"

#ifndef NODE_ID
#define NODE_ID 1
#endif

static_assert(NODE_ID >= 1 && NODE_ID <= 255, "NODE_ID must be in [1,255]");

namespace {

constexpr i2s_port_t I2S_PORT = I2S_NUM_0;
constexpr uint32_t SAMPLE_RATE = 16000;
constexpr size_t CHUNK_SIZE = 1024;
constexpr size_t FFT_SIZE = 1024;
constexpr size_t MFCC_FFT_SIZE = 512;
constexpr uint32_t MAG_PRINT_PERIOD_MS = 100;
constexpr uint32_t AUDIO_FLOW_PRINT_PERIOD_MS = 1000;
constexpr uint32_t PERF_PRINT_PERIOD_MS = 1000;
constexpr float TINYML_STD_EPS = 1e-6f;
constexpr int TINYML_MEL_BANDS = 13;

// Simulation-aligned defaults
constexpr float STA_WINDOW_MS = 50.0f;
constexpr float LTA_WINDOW_MS = 5000.0f;
constexpr float STA_LTA_THRESHOLD = 2.5f;
constexpr uint32_t COOLDOWN_MS = 2500;
constexpr bool FFT_PREFILTER_ENABLED = true;
constexpr uint16_t WHISTLE_FREQ_MIN_HZ = 2500;
constexpr uint16_t WHISTLE_FREQ_MAX_HZ = 4000;
constexpr bool WHISTLE_RESCUE_ENABLED = true;
constexpr uint16_t WHISTLE_RESCUE_MIN_PEAK = 280;
constexpr float BATTERY_DRAIN_PER_EVENT = 0.1f;

constexpr uint32_t STA_SAMPLES = static_cast<uint32_t>((STA_WINDOW_MS / 1000.0f) * SAMPLE_RATE);
constexpr uint32_t LTA_SAMPLES = static_cast<uint32_t>((LTA_WINDOW_MS / 1000.0f) * SAMPLE_RATE);
constexpr float ALPHA_STA = (STA_SAMPLES > 0) ? (2.0f / (static_cast<float>(STA_SAMPLES) + 1.0f)) : 0.05f;
constexpr float ALPHA_LTA = (LTA_SAMPLES > 0) ? (2.0f / (static_cast<float>(LTA_SAMPLES) + 1.0f)) : 0.0005f;
constexpr uint32_t COOLDOWN_SAMPLES = (COOLDOWN_MS * SAMPLE_RATE) / 1000;

// Audio input pins (INMP441 style I2S mic). Override with -D flags if needed.
#ifndef PIN_I2S_BCLK
#define PIN_I2S_BCLK 4
#endif

#ifndef PIN_I2S_WS
#define PIN_I2S_WS 5
#endif

#ifndef PIN_I2S_SD
#define PIN_I2S_SD 6
#endif

#ifndef I2S_USE_RIGHT_CHANNEL
#define I2S_USE_RIGHT_CHANNEL 0
#endif

// E22-900T22D UART pins. Override with -D flags if needed.
#ifndef PIN_LORA_TX
#define PIN_LORA_TX 8
#endif

#ifndef PIN_LORA_RX
#define PIN_LORA_RX 13
#endif

#ifndef PIN_LORA_AUX
#define PIN_LORA_AUX 11
#endif

#ifndef PIN_LORA_M0
#define PIN_LORA_M0 10
#endif

#ifndef PIN_LORA_M1
#define PIN_LORA_M1 9
#endif

constexpr uint32_t LORA_UART_BAUD = 9600;
constexpr int8_t DEFAULT_TEMP_C = 22;
constexpr uint8_t DEFAULT_GPS_HDOP = 15;

#ifndef PIN_GPS_TX
#define PIN_GPS_TX -1
#endif

#ifndef PIN_GPS_RX
#define PIN_GPS_RX -1
#endif

constexpr uint32_t GPS_UART_BAUD = 9600;
constexpr bool GPS_MOCK_ENABLED = true;
constexpr uint32_t GPS_MOCK_PERIOD_MS = 1000;
constexpr uint32_t GPS_RAW_TIMEOUT_MS = 4000;
constexpr double GPS_MOCK_BASE_LAT = 39.867000;
constexpr double GPS_MOCK_BASE_LON = 32.733000;
constexpr uint64_t GPS_MOCK_BASE_EPOCH_SEC = 1767225600ULL;  // 2026-01-01T00:00:00Z

constexpr uint8_t CLASS_WHISTLE = 0;
constexpr uint8_t CLASS_HUMAN_VOICE = 1;
constexpr uint8_t CLASS_IMPACT = 2;
constexpr uint8_t CLASS_KNOCKING = 3;
constexpr uint8_t CLASS_COLLAPSE = 4;
constexpr uint8_t CLASS_MACHINERY = 5;
constexpr uint8_t CLASS_MOTOR = 6;
constexpr uint8_t CLASS_ANIMAL = 7;
constexpr uint8_t CLASS_WIND = 8;
constexpr uint8_t CLASS_RAIN = 9;
constexpr uint8_t CLASS_AMBIENT = 10;

enum class PacketAction : uint8_t {
  Target,
  LogOnly,
  Reject,
};

struct DetectionResult {
  bool detected = false;
  uint16_t peakAmplitude = 0;
  uint8_t snrDb = 0;
  size_t triggerLen = 0;
  int16_t triggerRegion[CHUNK_SIZE] = {0};
};

struct FFTResult {
  uint16_t peakFreqHz = 0;
  uint16_t globalPeakFreqHz = 0;
  float peakMagnitude = 0.0f;
  bool inTargetBand = false;
  bool passed = false;
};

struct ClassResult {
  uint8_t classId = CLASS_AMBIENT;
  uint8_t confidencePct = 50;
  bool isTarget = false;
  PacketAction action = PacketAction::Reject;
};

struct AudioMetrics {
  uint16_t peak = 0;
  float rms = 0.0f;
  int16_t minSample = 0;
  int16_t maxSample = 0;
  float dc = 0.0f;
  uint32_t fingerprint = 0;
};

struct AudioFlowState {
  uint32_t chunksWindow = 0;
  uint32_t changedWindow = 0;
  uint32_t quietWindow = 0;
  uint32_t repeatingWindow = 0;
  uint32_t lastPrintMs = 0;
  uint32_t lastFingerprint = 0;
};

struct PerfTimings {
  uint32_t gpsUs = 0;
  uint32_t captureUs = 0;
  uint32_t convertUs = 0;
  uint32_t metricsUs = 0;
  uint32_t detectUs = 0;
  uint32_t fftUs = 0;
  uint32_t classifyUs = 0;
  uint32_t rescueUs = 0;
  uint32_t packetBuildUs = 0;
  uint32_t txUs = 0;
  uint32_t totalUs = 0;
  bool detected = false;
  bool txAttempted = false;
  bool txOk = false;
};

struct GpsState {
  bool rawUartEnabled = false;
  bool haveFix = false;
  bool fixFromRaw = false;
  bool haveDate = false;
  bool haveUtcTime = false;
  bool haveEpochAnchor = false;
  bool epochFromRaw = false;
  double latitude = GPS_MOCK_BASE_LAT;
  double longitude = GPS_MOCK_BASE_LON;
  float hdop = static_cast<float>(DEFAULT_GPS_HDOP) / 10.0f;
  double utcSecondsOfDay = 0.0;
  double lastUtcSecondsOfDay = -1.0;
  int year = 1970;
  int month = 1;
  int day = 1;
  uint8_t satellites = 0;
  uint64_t epochAnchorMicros = 0;
  int64_t monoAnchorMicros = 0;
  uint32_t lastUpdateMs = 0;
  uint32_t lastMockUpdateMs = 0;
  uint32_t lastStatusPrintMs = 0;
  char lineBuffer[128] = {0};
  size_t lineLength = 0;
};

#pragma pack(push, 1)
struct NodePacket {
  uint8_t node_id;
  uint64_t ts_micros;
  uint16_t magnitude;
  uint16_t peak_freq_hz;
  uint8_t ml_class;
  uint8_t ml_confidence;
  uint8_t snr_db;
  uint8_t battery_pct;
  int8_t temperature_c;
  uint8_t gps_hdop;
  int32_t latitude_e7;
  int32_t longitude_e7;
  uint16_t event_count;
  uint8_t crc8;
  uint8_t reserved;
};
#pragma pack(pop)

static_assert(sizeof(NodePacket) == 31, "NodePacket must be 31 bytes");
constexpr size_t NODE_PACKET_CRC_LEN = sizeof(NodePacket) - 2;

int32_t gAudioRawChunk[CHUNK_SIZE] = {0};
int16_t gAudioChunk[CHUNK_SIZE] = {0};
float gFFTReal[FFT_SIZE] = {0.0f};
float gFFTImag[FFT_SIZE] = {0.0f};
float gMagnitudes[(FFT_SIZE / 2) + 1] = {0.0f};
float gTinyReal[FFT_SIZE] = {0.0f};
float gTinyImag[FFT_SIZE] = {0.0f};
float gTinySpectrum[(FFT_SIZE / 2) + 1] = {0.0f};
float gTinyMelFilters[TINYML_MEL_BANDS][(MFCC_FFT_SIZE / 2) + 1] = {{0.0f}};
bool gTinyMelReady = false;
float gTinyFeatures[kTinyMlInputDim] = {0.0f};
float gTinyInput[kTinyMlInputDim] = {0.0f};
float gTinyHidden1[kTinyMlHidden1Dim] = {0.0f};
float gTinyHidden2[kTinyMlHidden2Dim] = {0.0f};
float gTinyLogits[kTinyMlOutputDim] = {0.0f};
float gTinyProbs[kTinyMlOutputDim] = {0.0f};

float gSta = 0.0f;
float gLta = 1.0f;
uint32_t gSamplesSinceTrigger = COOLDOWN_SAMPLES;
uint16_t gEventCount = 0;
float gBatteryPct = 100.0f;
GpsState gGps{};
AudioFlowState gAudioFlow{};

void fftInPlace(float* real, float* imag, const size_t n);

const char* className(const uint8_t classId) {
  switch (classId) {
    case CLASS_WHISTLE:
      return "whistle";
    case CLASS_HUMAN_VOICE:
      return "human_voice";
    case CLASS_IMPACT:
      return "impact";
    case CLASS_KNOCKING:
      return "knocking";
    case CLASS_COLLAPSE:
      return "collapse";
    case CLASS_MACHINERY:
      return "machinery";
    case CLASS_MOTOR:
      return "motor";
    case CLASS_ANIMAL:
      return "animal";
    case CLASS_WIND:
      return "wind";
    case CLASS_RAIN:
      return "rain";
    default:
      return "ambient";
  }
}

template <typename T>
T clampValue(const T value, const T minValue, const T maxValue) {
  if (value < minValue) {
    return minValue;
  }
  if (value > maxValue) {
    return maxValue;
  }
  return value;
}

float hzToMel(const float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }

float melToHz(const float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

void initTinyMlMelFilters() {
  if (gTinyMelReady) {
    return;
  }

  const int nMels = TINYML_MEL_BANDS;
  const int nFft = static_cast<int>(MFCC_FFT_SIZE);
  const int nBins = nFft / 2 + 1;

  const float melMin = 0.0f;
  const float melMax = hzToMel(static_cast<float>(SAMPLE_RATE) * 0.5f);

  float melPoints[TINYML_MEL_BANDS + 2] = {0.0f};
  int binPoints[TINYML_MEL_BANDS + 2] = {0};

  for (int i = 0; i < nMels + 2; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(nMels + 1);
    melPoints[i] = melMin + t * (melMax - melMin);
    const float hz = melToHz(melPoints[i]);
    int bin = static_cast<int>(floorf((static_cast<float>(nFft) + 1.0f) * hz / static_cast<float>(SAMPLE_RATE)));
    bin = clampValue(bin, 0, nBins - 1);
    binPoints[i] = bin;
  }

  memset(gTinyMelFilters, 0, sizeof(gTinyMelFilters));
  for (int m = 0; m < nMels; ++m) {
    const int fStart = binPoints[m];
    const int fCenter = binPoints[m + 1];
    const int fEnd = binPoints[m + 2];

    if (fCenter > fStart) {
      for (int k = fStart; k < fCenter; ++k) {
        gTinyMelFilters[m][k] = static_cast<float>(k - fStart) / static_cast<float>(fCenter - fStart);
      }
    }
    if (fEnd > fCenter) {
      for (int k = fCenter; k < fEnd; ++k) {
        gTinyMelFilters[m][k] = static_cast<float>(fEnd - k) / static_cast<float>(fEnd - fCenter);
      }
    }
  }

  gTinyMelReady = true;
}

void computeRfftMagnitudeFromI16(const int16_t* samples, const size_t length, const size_t nfft, float* outMagnitude,
                                 const size_t outBins) {
  if (samples == nullptr || outMagnitude == nullptr || nfft == 0 || nfft > FFT_SIZE) {
    return;
  }

  memset(gTinyReal, 0, sizeof(gTinyReal));
  memset(gTinyImag, 0, sizeof(gTinyImag));

  const size_t copyLen = (length < nfft) ? length : nfft;
  for (size_t i = 0; i < copyLen; ++i) {
    gTinyReal[i] = static_cast<float>(samples[i]);
  }

  float mean = 0.0f;
  for (size_t i = 0; i < nfft; ++i) {
    mean += gTinyReal[i];
  }
  mean /= static_cast<float>(nfft);

  for (size_t i = 0; i < nfft; ++i) {
    const float window = 0.54f - 0.46f * cosf((2.0f * PI * static_cast<float>(i)) / static_cast<float>(nfft - 1));
    gTinyReal[i] = (gTinyReal[i] - mean) * window;
  }

  fftInPlace(gTinyReal, gTinyImag, nfft);

  const size_t bins = nfft / 2 + 1;
  const size_t calcBins = (outBins < bins) ? outBins : bins;
  for (size_t k = 0; k < calcBins; ++k) {
    const float re = gTinyReal[k];
    const float im = gTinyImag[k];
    outMagnitude[k] = sqrtf(re * re + im * im);
  }
  for (size_t k = calcBins; k < outBins; ++k) {
    outMagnitude[k] = 0.0f;
  }
}

void computeSegmentMfcc13(const int16_t* segment, const size_t len, float outMfcc[TINYML_MEL_BANDS]) {
  if (outMfcc == nullptr) {
    return;
  }
  for (int i = 0; i < TINYML_MEL_BANDS; ++i) {
    outMfcc[i] = 0.0f;
  }

  if (segment == nullptr || len < 16) {
    return;
  }

  computeRfftMagnitudeFromI16(segment, len, MFCC_FFT_SIZE, gTinySpectrum, (MFCC_FFT_SIZE / 2) + 1);

  for (int m = 0; m < TINYML_MEL_BANDS; ++m) {
    float melEnergy = 0.0f;
    for (size_t k = 0; k <= (MFCC_FFT_SIZE / 2); ++k) {
      melEnergy += gTinyMelFilters[m][k] * gTinySpectrum[k];
    }
    outMfcc[m] = logf(melEnergy + 1e-10f);
  }
}

float computeRms(const int16_t* samples, const size_t length) {
  if (length == 0) {
    return 0.0f;
  }
  double sumSquares = 0.0;
  for (size_t i = 0; i < length; ++i) {
    const float v = static_cast<float>(samples[i]);
    sumSquares += static_cast<double>(v) * static_cast<double>(v);
  }
  return sqrtf(static_cast<float>(sumSquares / static_cast<double>(length)));
}

float computeZcr(const int16_t* samples, const size_t length) {
  if (length < 2) {
    return 0.0f;
  }
  uint32_t crossingAccum = 0;
  for (size_t i = 1; i < length; ++i) {
    const int sPrev = (samples[i - 1] > 0) - (samples[i - 1] < 0);
    const int sCurr = (samples[i] > 0) - (samples[i] < 0);
    crossingAccum += static_cast<uint32_t>(abs(sCurr - sPrev));
  }
  return static_cast<float>(crossingAccum) / (2.0f * static_cast<float>(length));
}

float computeCrestFactor(const int16_t* samples, const size_t length, const float rms) {
  if (length == 0 || rms <= 1e-6f) {
    return 0.0f;
  }
  uint16_t peak = 0;
  for (size_t i = 0; i < length; ++i) {
    const uint16_t v = static_cast<uint16_t>(abs(static_cast<int32_t>(samples[i])));
    if (v > peak) {
      peak = v;
    }
  }
  return static_cast<float>(peak) / rms;
}

float computeEnvelopeVariance(const int16_t* samples, const size_t length, const float rms) {
  if (length < 8 || rms <= 1e-6f) {
    return 0.0f;
  }

  constexpr size_t kSubWindows = 4;
  const size_t subSize = length / kSubWindows;
  if (subSize == 0) {
    return 0.0f;
  }

  float subRms[kSubWindows] = {0.0f};
  for (size_t s = 0; s < kSubWindows; ++s) {
    const size_t begin = s * subSize;
    const size_t end = (s == (kSubWindows - 1)) ? length : (begin + subSize);
    if (end <= begin) {
      continue;
    }
    subRms[s] = computeRms(samples + begin, end - begin);
  }

  float mean = 0.0f;
  for (float value : subRms) {
    mean += value;
  }
  mean /= static_cast<float>(kSubWindows);

  float variance = 0.0f;
  for (float value : subRms) {
    const float diff = value - mean;
    variance += diff * diff;
  }
  variance /= static_cast<float>(kSubWindows);

  return variance / (rms * rms + 1e-10f);
}

bool extractTinyMlFeatures(const int16_t* audio, const size_t len, float outFeatures[kTinyMlInputDim]) {
  if (audio == nullptr || outFeatures == nullptr || len < 128) {
    return false;
  }

  const size_t nfft = (len < FFT_SIZE) ? len : FFT_SIZE;
  const size_t bins = nfft / 2 + 1;

  computeRfftMagnitudeFromI16(audio, nfft, nfft, gTinySpectrum, bins);

  const float binWidth = static_cast<float>(SAMPLE_RATE) / static_cast<float>(nfft);
  float specSum = 0.0f;
  float weightedFreqSum = 0.0f;
  float weightedBwSum = 0.0f;
  float logSum = 0.0f;
  float arithSum = 0.0f;

  for (size_t k = 0; k < bins; ++k) {
    const float mag = gTinySpectrum[k];
    const float freq = static_cast<float>(k) * binWidth;
    specSum += mag;
    weightedFreqSum += freq * mag;
    logSum += logf(mag + 1e-10f);
    arithSum += mag;
  }

  float centroidHz = 0.0f;
  float bandwidthHz = 0.0f;
  float flatness = 0.0f;
  float rolloffHz = 0.0f;
  float skewness = 0.0f;

  if (specSum > 0.0f) {
    centroidHz = weightedFreqSum / specSum;

    for (size_t k = 0; k < bins; ++k) {
      const float freq = static_cast<float>(k) * binWidth;
      const float diff = freq - centroidHz;
      weightedBwSum += diff * diff * gTinySpectrum[k];
    }
    bandwidthHz = sqrtf(weightedBwSum / specSum);

    float cumsum = 0.0f;
    const float rolloffTarget = 0.85f * specSum;
    size_t rolloffIdx = 0;
    for (size_t k = 0; k < bins; ++k) {
      cumsum += gTinySpectrum[k];
      if (cumsum >= rolloffTarget) {
        rolloffIdx = k;
        break;
      }
    }
    rolloffHz = static_cast<float>(rolloffIdx) * binWidth;

    const float geomMean = expf(logSum / static_cast<float>(bins));
    const float arithMean = arithSum / static_cast<float>(bins);
    flatness = geomMean / (arithMean + 1e-10f);

    float m3 = 0.0f;
    for (size_t k = 0; k < bins; ++k) {
      const float freq = static_cast<float>(k) * binWidth;
      const float diff = freq - centroidHz;
      m3 += diff * diff * diff * gTinySpectrum[k];
    }
    m3 /= specSum;
    skewness = m3 / (bandwidthHz * bandwidthHz * bandwidthHz + 1e-10f);
  }

  const float rms = computeRms(audio, len);
  const float zcr = computeZcr(audio, len);
  const float crestFactor = computeCrestFactor(audio, len, rms);
  const float envVariance = computeEnvelopeVariance(audio, len, rms);

  const size_t half = len / 2;
  float mfccH1[TINYML_MEL_BANDS] = {0.0f};
  float mfccH2[TINYML_MEL_BANDS] = {0.0f};
  computeSegmentMfcc13(audio, half, mfccH1);
  computeSegmentMfcc13(audio + half, len - half, mfccH2);

  for (int i = 0; i < TINYML_MEL_BANDS; ++i) {
    outFeatures[i] = 0.5f * (mfccH1[i] + mfccH2[i]);
  }

  outFeatures[13] = rms / 32768.0f;
  outFeatures[14] = zcr;
  outFeatures[15] = centroidHz / (static_cast<float>(SAMPLE_RATE) * 0.5f);
  outFeatures[16] = bandwidthHz / (static_cast<float>(SAMPLE_RATE) * 0.5f);
  outFeatures[17] = rolloffHz / (static_cast<float>(SAMPLE_RATE) * 0.5f);
  outFeatures[18] = flatness;

  for (int i = 0; i < TINYML_MEL_BANDS; ++i) {
    outFeatures[19 + i] = mfccH2[i] - mfccH1[i];
  }

  outFeatures[32] = crestFactor / 10.0f;
  outFeatures[33] = envVariance;
  outFeatures[34] = skewness / 5.0f;
  return true;
}

void denseLayer(const float* input, const int inDim, const float* weights, const float* bias, const int outDim,
                const bool reluEnabled, float* output) {
  for (int j = 0; j < outDim; ++j) {
    float acc = bias[j];
    for (int i = 0; i < inDim; ++i) {
      acc += input[i] * weights[i * outDim + j];
    }
    output[j] = reluEnabled ? ((acc > 0.0f) ? acc : 0.0f) : acc;
  }
}

void softmax(const float* logits, const int dim, float* probs) {
  float maxLogit = logits[0];
  for (int i = 1; i < dim; ++i) {
    if (logits[i] > maxLogit) {
      maxLogit = logits[i];
    }
  }

  float sumExp = 0.0f;
  for (int i = 0; i < dim; ++i) {
    probs[i] = expf(logits[i] - maxLogit);
    sumExp += probs[i];
  }

  const float invSum = (sumExp > 0.0f) ? (1.0f / sumExp) : 1.0f;
  for (int i = 0; i < dim; ++i) {
    probs[i] *= invSum;
  }
}

void printInitStep(const char* step, const char* message) {
  Serial.printf("[INIT] %-12s %s\n", step, message);
}

int hexNibble(const char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  }
  if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  }
  return -1;
}

bool validateNmeaChecksum(const char* sentence) {
  if (sentence == nullptr || sentence[0] != '$') {
    return false;
  }

  const char* star = strchr(sentence, '*');
  if (star == nullptr || (star - sentence) < 2) {
    return false;
  }
  if (star[1] == '\0' || star[2] == '\0') {
    return false;
  }

  uint8_t checksum = 0;
  for (const char* p = sentence + 1; p < star; ++p) {
    checksum ^= static_cast<uint8_t>(*p);
  }

  const int hi = hexNibble(star[1]);
  const int lo = hexNibble(star[2]);
  if (hi < 0 || lo < 0) {
    return false;
  }

  const uint8_t parsedChecksum = static_cast<uint8_t>((hi << 4) | lo);
  return checksum == parsedChecksum;
}

double nmeaCoordToDecimal(const char* token, const char hemi) {
  if (token == nullptr || token[0] == '\0') {
    return 0.0;
  }

  const double raw = strtod(token, nullptr);
  const double degrees = floor(raw / 100.0);
  const double minutes = raw - (degrees * 100.0);
  double decimal = degrees + (minutes / 60.0);

  if (hemi == 'S' || hemi == 'W') {
    decimal = -decimal;
  }
  return decimal;
}

void decimalToNmeaCoord(const double value, const bool isLat, char* outCoord, const size_t outSize,
                        char& outHemi) {
  const double absValue = fabs(value);
  const int degrees = static_cast<int>(floor(absValue));
  const double minutes = (absValue - static_cast<double>(degrees)) * 60.0;

  outHemi = isLat ? ((value >= 0.0) ? 'N' : 'S') : ((value >= 0.0) ? 'E' : 'W');

  if (isLat) {
    snprintf(outCoord, outSize, "%02d%07.4f", degrees, minutes);
  } else {
    snprintf(outCoord, outSize, "%03d%07.4f", degrees, minutes);
  }
}

int64_t daysFromCivil(int y, int m, int d) {
  y -= (m <= 2) ? 1 : 0;
  const int era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(y - era * 400);
  const unsigned doy = (153U * static_cast<unsigned>(m + (m > 2 ? -3 : 9)) + 2U) / 5U + static_cast<unsigned>(d) - 1U;
  const unsigned doe = yoe * 365U + yoe / 4U - yoe / 100U + doy;
  return static_cast<int64_t>(era) * 146097LL + static_cast<int64_t>(doe) - 719468LL;
}

void civilFromDays(int64_t z, int& y, int& m, int& d) {
  z += 719468;
  const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = static_cast<unsigned>(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  y = static_cast<int>(yoe) + static_cast<int>(era) * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  d = static_cast<int>(doy - (153 * mp + 2) / 5 + 1);
  m = static_cast<int>(mp + (mp < 10 ? 3 : -9));
  y += (m <= 2) ? 1 : 0;
}

bool parseUtcTokenSeconds(const char* token, double& outSeconds) {
  outSeconds = 0.0;
  if (token == nullptr || token[0] == '\0') {
    return false;
  }

  const double raw = strtod(token, nullptr);
  const int hh = static_cast<int>(raw / 10000.0);
  const int mm = static_cast<int>(raw / 100.0) % 100;
  const double ss = raw - static_cast<double>(hh * 10000 + mm * 100);

  if (hh < 0 || hh > 23 || mm < 0 || mm > 59 || ss < 0.0 || ss >= 60.0) {
    return false;
  }

  outSeconds = static_cast<double>(hh * 3600 + mm * 60) + ss;
  return true;
}

bool parseDateTokenDdMmYy(const char* token, int& outYear, int& outMonth, int& outDay) {
  outYear = 1970;
  outMonth = 1;
  outDay = 1;

  if (token == nullptr || strlen(token) < 6) {
    return false;
  }

  for (int i = 0; i < 6; ++i) {
    if (token[i] < '0' || token[i] > '9') {
      return false;
    }
  }

  const int dd = (token[0] - '0') * 10 + (token[1] - '0');
  const int mm = (token[2] - '0') * 10 + (token[3] - '0');
  const int yy = (token[4] - '0') * 10 + (token[5] - '0');

  if (dd < 1 || dd > 31 || mm < 1 || mm > 12 || yy < 0 || yy > 99) {
    return false;
  }

  outYear = 2000 + yy;
  outMonth = mm;
  outDay = dd;
  return true;
}

void incrementGpsDateByDays(const int deltaDays) {
  if (!gGps.haveDate || deltaDays == 0) {
    return;
  }
  const int64_t days = daysFromCivil(gGps.year, gGps.month, gGps.day) + static_cast<int64_t>(deltaDays);
  civilFromDays(days, gGps.year, gGps.month, gGps.day);
}

uint64_t epochMicrosFromUtc(const int year, const int month, const int day, const double utcSecondsOfDay) {
  const int64_t days = daysFromCivil(year, month, day);
  const int64_t wholeSec = static_cast<int64_t>(floor(utcSecondsOfDay));
  const double frac = utcSecondsOfDay - static_cast<double>(wholeSec);
  int64_t microsFrac = static_cast<int64_t>(llround(frac * 1000000.0));
  int64_t epochSec = days * 86400LL + wholeSec;

  if (microsFrac >= 1000000LL) {
    epochSec += microsFrac / 1000000LL;
    microsFrac %= 1000000LL;
  } else if (microsFrac < 0) {
    const int64_t borrow = ((-microsFrac) + 999999LL) / 1000000LL;
    epochSec -= borrow;
    microsFrac += borrow * 1000000LL;
  }

  return static_cast<uint64_t>(epochSec * 1000000LL + microsFrac);
}

void updateGpsEpochAnchor(const bool fromRaw) {
  if (!gGps.haveDate || !gGps.haveUtcTime) {
    return;
  }

  gGps.epochAnchorMicros = epochMicrosFromUtc(gGps.year, gGps.month, gGps.day, gGps.utcSecondsOfDay);
  gGps.monoAnchorMicros = static_cast<int64_t>(esp_timer_get_time());
  gGps.haveEpochAnchor = true;
  gGps.epochFromRaw = fromRaw;
}

uint64_t currentGpsEpochMicros() {
  if (gGps.haveEpochAnchor) {
    const int64_t nowUs = static_cast<int64_t>(esp_timer_get_time());
    const int64_t delta = nowUs - gGps.monoAnchorMicros;
    const int64_t clampedDelta = (delta > 0) ? delta : 0;
    return gGps.epochAnchorMicros + static_cast<uint64_t>(clampedDelta);
  }

  // Fallback to mock GPS epoch baseline until first UTC+date lock.
  return GPS_MOCK_BASE_EPOCH_SEC * 1000000ULL + static_cast<uint64_t>(millis()) * 1000ULL;
}

void setGpsFix(const double lat, const double lon, const float hdop, const uint8_t sats, const bool fromRaw) {
  gGps.haveFix = true;
  gGps.fixFromRaw = fromRaw;
  gGps.latitude = lat;
  gGps.longitude = lon;
  gGps.hdop = (hdop > 0.0f) ? hdop : (static_cast<float>(DEFAULT_GPS_HDOP) / 10.0f);
  gGps.satellites = sats;
  gGps.lastUpdateMs = millis();
}

bool parseGgaFields(char* sentenceMutable, double& outUtcSeconds, double& outLat, double& outLon, float& outHdop,
                    uint8_t& outSats) {
  char* star = strchr(sentenceMutable, '*');
  if (star != nullptr) {
    *star = '\0';
  }

  char* fields[20] = {nullptr};
  size_t fieldCount = 0;
  fields[fieldCount++] = sentenceMutable;
  for (char* p = sentenceMutable; *p != '\0' && fieldCount < 20; ++p) {
    if (*p == ',') {
      *p = '\0';
      fields[fieldCount++] = p + 1;
    }
  }

  if (fieldCount < 9) {
    return false;
  }

  if (!parseUtcTokenSeconds(fields[1], outUtcSeconds)) {
    return false;
  }

  const int fixQuality = atoi(fields[6]);
  if (fixQuality <= 0) {
    return false;
  }

  const char latHemi = (fields[3] != nullptr && fields[3][0] != '\0') ? fields[3][0] : 'N';
  const char lonHemi = (fields[5] != nullptr && fields[5][0] != '\0') ? fields[5][0] : 'E';

  outLat = nmeaCoordToDecimal(fields[2], latHemi);
  outLon = nmeaCoordToDecimal(fields[4], lonHemi);
  outHdop = (fields[8] != nullptr && fields[8][0] != '\0') ? strtof(fields[8], nullptr)
                                                             : (static_cast<float>(DEFAULT_GPS_HDOP) / 10.0f);
  outSats = static_cast<uint8_t>(clampValue(atoi(fields[7]), 0, 99));
  return true;
}

bool parseRmcFields(char* sentenceMutable, bool& outActiveFix, double& outUtcSeconds, int& outYear, int& outMonth,
                    int& outDay, bool& outHasCoords, double& outLat, double& outLon) {
  char* star = strchr(sentenceMutable, '*');
  if (star != nullptr) {
    *star = '\0';
  }

  char* fields[20] = {nullptr};
  size_t fieldCount = 0;
  fields[fieldCount++] = sentenceMutable;
  for (char* p = sentenceMutable; *p != '\0' && fieldCount < 20; ++p) {
    if (*p == ',') {
      *p = '\0';
      fields[fieldCount++] = p + 1;
    }
  }

  if (fieldCount < 10) {
    return false;
  }

  if (!parseUtcTokenSeconds(fields[1], outUtcSeconds)) {
    return false;
  }
  if (!parseDateTokenDdMmYy(fields[9], outYear, outMonth, outDay)) {
    return false;
  }

  outActiveFix = (fields[2] != nullptr && fields[2][0] == 'A');
  outHasCoords = false;
  outLat = 0.0;
  outLon = 0.0;

  if (fields[3] != nullptr && fields[3][0] != '\0' && fields[4] != nullptr && fields[4][0] != '\0' &&
      fields[5] != nullptr && fields[5][0] != '\0' && fields[6] != nullptr && fields[6][0] != '\0') {
    outLat = nmeaCoordToDecimal(fields[3], fields[4][0]);
    outLon = nmeaCoordToDecimal(fields[5], fields[6][0]);
    outHasCoords = true;
  }

  return true;
}

void parseGpsLine(const char* line, const bool fromRaw) {
  if (line == nullptr || line[0] == '\0' || line[0] != '$') {
    return;
  }

  const bool isGga = (strncmp(line, "$GPGGA", 6) == 0) || (strncmp(line, "$GNGGA", 6) == 0);
  const bool isRmc = (strncmp(line, "$GPRMC", 6) == 0) || (strncmp(line, "$GNRMC", 6) == 0);
  if (!isGga && !isRmc) {
    return;
  }

  if (!validateNmeaChecksum(line)) {
    return;
  }

  char copy[128] = {0};
  strncpy(copy, line, sizeof(copy) - 1);

  if (isGga) {
    double utcSeconds = 0.0;
    double lat = 0.0;
    double lon = 0.0;
    float hdop = static_cast<float>(DEFAULT_GPS_HDOP) / 10.0f;
    uint8_t sats = 0;
    if (!parseGgaFields(copy, utcSeconds, lat, lon, hdop, sats)) {
      return;
    }

    if (gGps.haveDate && gGps.lastUtcSecondsOfDay >= 0.0 && (utcSeconds + 43200.0) < gGps.lastUtcSecondsOfDay) {
      incrementGpsDateByDays(1);
    }

    gGps.haveUtcTime = true;
    gGps.utcSecondsOfDay = utcSeconds;
    gGps.lastUtcSecondsOfDay = utcSeconds;

    setGpsFix(lat, lon, hdop, sats, fromRaw);
    updateGpsEpochAnchor(fromRaw);
    return;
  }

  bool activeFix = false;
  double utcSeconds = 0.0;
  int year = 1970;
  int month = 1;
  int day = 1;
  bool hasCoords = false;
  double lat = 0.0;
  double lon = 0.0;
  if (!parseRmcFields(copy, activeFix, utcSeconds, year, month, day, hasCoords, lat, lon)) {
    return;
  }

  gGps.haveDate = true;
  gGps.year = year;
  gGps.month = month;
  gGps.day = day;
  gGps.haveUtcTime = true;
  gGps.utcSecondsOfDay = utcSeconds;
  gGps.lastUtcSecondsOfDay = utcSeconds;

  if (activeFix && hasCoords) {
    const float hdop = (gGps.hdop > 0.0f) ? gGps.hdop : (static_cast<float>(DEFAULT_GPS_HDOP) / 10.0f);
    setGpsFix(lat, lon, hdop, gGps.satellites, fromRaw);
  }

  updateGpsEpochAnchor(fromRaw);
}

void pollRawGpsSignal() {
  if (!gGps.rawUartEnabled) {
    return;
  }

  while (Serial1.available() > 0) {
    const char c = static_cast<char>(Serial1.read());

    if (c == '\r') {
      continue;
    }

    if (c == '\n') {
      if (gGps.lineLength > 6) {
        gGps.lineBuffer[gGps.lineLength] = '\0';
        parseGpsLine(gGps.lineBuffer, true);
      }
      gGps.lineLength = 0;
      continue;
    }

    if (gGps.lineLength < (sizeof(gGps.lineBuffer) - 1)) {
      gGps.lineBuffer[gGps.lineLength++] = c;
    } else {
      gGps.lineLength = 0;
    }
  }
}

void pushMockGpsSentence() {
  const uint32_t now = millis();
  if (!GPS_MOCK_ENABLED || (now - gGps.lastMockUpdateMs) < GPS_MOCK_PERIOD_MS) {
    return;
  }

  gGps.lastMockUpdateMs = now;

  const double t = static_cast<double>(now) / 1000.0;
  const double lat = GPS_MOCK_BASE_LAT + 0.00003 * sin(t * 0.10);
  const double lon = GPS_MOCK_BASE_LON + 0.00003 * cos(t * 0.08);
  const float hdop = 0.9f + 0.5f * (0.5f + 0.5f * sinf(static_cast<float>(t * 0.12)));
  const uint8_t sats = static_cast<uint8_t>(8 + ((now / 1000) % 4));

  const uint64_t nowEpochSec = GPS_MOCK_BASE_EPOCH_SEC + static_cast<uint64_t>(now) / 1000ULL;
  const uint32_t secOfDay = static_cast<uint32_t>(nowEpochSec % 86400ULL);
  const int64_t epochDays = static_cast<int64_t>(nowEpochSec / 86400ULL);
  int year = 1970;
  int month = 1;
  int day = 1;
  civilFromDays(epochDays, year, month, day);

  const uint32_t hh = secOfDay / 3600;
  const uint32_t mm = (secOfDay % 3600) / 60;
  const uint32_t ss = secOfDay % 60;

  char utc[12] = {0};
  snprintf(utc, sizeof(utc), "%02u%02u%02u.00", static_cast<unsigned>(hh), static_cast<unsigned>(mm),
           static_cast<unsigned>(ss));

  char date[8] = {0};
  snprintf(date, sizeof(date), "%02d%02d%02d", day, month, year % 100);

  char latToken[16] = {0};
  char lonToken[16] = {0};
  char latHemi = 'N';
  char lonHemi = 'E';
  decimalToNmeaCoord(lat, true, latToken, sizeof(latToken), latHemi);
  decimalToNmeaCoord(lon, false, lonToken, sizeof(lonToken), lonHemi);

  char ggaBody[120] = {0};
  snprintf(ggaBody, sizeof(ggaBody), "GPGGA,%s,%s,%c,%s,%c,1,%02u,%.1f,0.0,M,0.0,M,,", utc, latToken, latHemi,
           lonToken, lonHemi, static_cast<unsigned>(sats), hdop);

  uint8_t checksum = 0;
  for (const char* p = ggaBody; *p != '\0'; ++p) {
    checksum ^= static_cast<uint8_t>(*p);
  }

  char ggaSentence[128] = {0};
  snprintf(ggaSentence, sizeof(ggaSentence), "$%s*%02X", ggaBody, static_cast<unsigned>(checksum));
  parseGpsLine(ggaSentence, false);

  char rmcBody[140] = {0};
  snprintf(rmcBody, sizeof(rmcBody), "GPRMC,%s,A,%s,%c,%s,%c,0.00,0.00,%s,,,A", utc, latToken, latHemi, lonToken,
           lonHemi, date);

  checksum = 0;
  for (const char* p = rmcBody; *p != '\0'; ++p) {
    checksum ^= static_cast<uint8_t>(*p);
  }

  char rmcSentence[160] = {0};
  snprintf(rmcSentence, sizeof(rmcSentence), "$%s*%02X", rmcBody, static_cast<unsigned>(checksum));
  parseGpsLine(rmcSentence, false);
}

void updateGps() {
  pollRawGpsSignal();

  const uint32_t now = millis();
  const bool rawFixFresh =
      gGps.haveFix && gGps.fixFromRaw && ((now - gGps.lastUpdateMs) <= GPS_RAW_TIMEOUT_MS);

  if (!rawFixFresh) {
    pushMockGpsSentence();
  }

  if ((now - gGps.lastStatusPrintMs) >= 1000) {
    gGps.lastStatusPrintMs = now;
    const char* mode = gGps.fixFromRaw ? "raw" : "mock";
    const char* epochMode = gGps.haveEpochAnchor ? (gGps.epochFromRaw ? "raw" : "mock") : "none";
    const uint64_t epochUs = currentGpsEpochMicros();
    if (gGps.haveFix) {
      Serial.printf(
          "GPS mode=%s epoch=%s ts=%llu lat=%.6f lon=%.6f hdop=%.1f sats=%u\n", mode, epochMode,
          static_cast<unsigned long long>(epochUs), gGps.latitude, gGps.longitude, gGps.hdop,
          static_cast<unsigned>(gGps.satellites));
    } else {
      Serial.printf("GPS mode=none epoch=%s ts=%llu waiting for fix\n", epochMode,
                    static_cast<unsigned long long>(epochUs));
    }
  }
}

uint8_t currentGpsHdopField() {
  const float hdop = gGps.haveFix ? gGps.hdop : (static_cast<float>(DEFAULT_GPS_HDOP) / 10.0f);
  const int scaled = static_cast<int>(roundf(hdop * 10.0f));
  return static_cast<uint8_t>(clampValue(scaled, 0, 255));
}

int32_t currentGpsLatitudeE7() {
  const double lat = gGps.haveFix ? gGps.latitude : GPS_MOCK_BASE_LAT;
  const int64_t scaled = static_cast<int64_t>(llround(lat * 10000000.0));
  return static_cast<int32_t>(clampValue<int64_t>(scaled, INT32_MIN, INT32_MAX));
}

int32_t currentGpsLongitudeE7() {
  const double lon = gGps.haveFix ? gGps.longitude : GPS_MOCK_BASE_LON;
  const int64_t scaled = static_cast<int64_t>(llround(lon * 10000000.0));
  return static_cast<int32_t>(clampValue<int64_t>(scaled, INT32_MIN, INT32_MAX));
}

uint8_t crc8(const uint8_t* data, const size_t length) {
  uint8_t crc = 0;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (uint8_t b = 0; b < 8; ++b) {
      if ((crc & 0x80U) != 0U) {
        crc = static_cast<uint8_t>((crc << 1) ^ 0x07);
      } else {
        crc <<= 1;
      }
    }
  }
  return crc;
}

uint32_t fingerprintAudio(const int16_t* samples, const size_t length) {
  uint32_t hash = 2166136261UL;
  for (size_t i = 0; i < length; i += 4) {
    hash ^= static_cast<uint16_t>(samples[i]);
    hash *= 16777619UL;
  }
  return hash;
}

bool isTargetClass(const uint8_t classId) {
  return classId == CLASS_WHISTLE || classId == CLASS_HUMAN_VOICE || classId == CLASS_IMPACT ||
         classId == CLASS_KNOCKING;
}

bool isLogOnlyClass(const uint8_t classId) { return classId == CLASS_COLLAPSE; }

bool waitForAuxReady(const uint32_t timeoutMs) {
  if (PIN_LORA_AUX < 0) {
    return true;
  }

  const uint32_t start = millis();
  while ((millis() - start) < timeoutMs) {
    if (digitalRead(PIN_LORA_AUX) == HIGH) {
      return true;
    }
    delay(1);
  }
  return false;
}

void fftInPlace(float* real, float* imag, const size_t n) {
  size_t j = 0;
  for (size_t i = 1; i < n; ++i) {
    size_t bit = n >> 1;
    while ((j & bit) != 0U) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      const float tr = real[i];
      const float ti = imag[i];
      real[i] = real[j];
      imag[i] = imag[j];
      real[j] = tr;
      imag[j] = ti;
    }
  }

  for (size_t len = 2; len <= n; len <<= 1) {
    const float angle = -2.0f * PI / static_cast<float>(len);
    const float wlenRe = cosf(angle);
    const float wlenIm = sinf(angle);

    for (size_t i = 0; i < n; i += len) {
      float wRe = 1.0f;
      float wIm = 0.0f;

      for (size_t k = 0; k < len / 2; ++k) {
        const size_t u = i + k;
        const size_t v = i + k + len / 2;

        const float vr = real[v] * wRe - imag[v] * wIm;
        const float vi = real[v] * wIm + imag[v] * wRe;
        const float ur = real[u];
        const float ui = imag[u];

        real[u] = ur + vr;
        imag[u] = ui + vi;
        real[v] = ur - vr;
        imag[v] = ui - vi;

        const float nextWRe = wRe * wlenRe - wIm * wlenIm;
        wIm = wRe * wlenIm + wIm * wlenRe;
        wRe = nextWRe;
      }
    }
  }
}

void initI2S() {
  i2s_config_t cfg = {};
  cfg.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate = SAMPLE_RATE;
  // INMP441 provides 24-bit audio packed in 32-bit I2S words.
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  cfg.channel_format = I2S_USE_RIGHT_CHANNEL ? I2S_CHANNEL_FMT_ONLY_RIGHT : I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = CHUNK_SIZE;
  cfg.use_apll = false;
  cfg.tx_desc_auto_clear = false;
  cfg.fixed_mclk = 0;

  const esp_err_t installErr = i2s_driver_install(I2S_PORT, &cfg, 0, nullptr);
  if (installErr != ESP_OK) {
    Serial.printf("I2S driver install failed: %d\n", static_cast<int>(installErr));
    while (true) {
      delay(1000);
    }
  }

  i2s_pin_config_t pins = {};
  pins.bck_io_num = PIN_I2S_BCLK;
  pins.ws_io_num = PIN_I2S_WS;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = PIN_I2S_SD;

  const esp_err_t pinErr = i2s_set_pin(I2S_PORT, &pins);
  if (pinErr != ESP_OK) {
    Serial.printf("I2S pin config failed: %d\n", static_cast<int>(pinErr));
    while (true) {
      delay(1000);
    }
  }

  i2s_zero_dma_buffer(I2S_PORT);
}

void initLoRaUart() {
  if (PIN_LORA_M0 >= 0) {
    pinMode(PIN_LORA_M0, OUTPUT);
    digitalWrite(PIN_LORA_M0, LOW);
  }
  if (PIN_LORA_M1 >= 0) {
    pinMode(PIN_LORA_M1, OUTPUT);
    digitalWrite(PIN_LORA_M1, LOW);
  }
  if (PIN_LORA_AUX >= 0) {
    pinMode(PIN_LORA_AUX, INPUT_PULLUP);
  }

  Serial2.begin(LORA_UART_BAUD, SERIAL_8N1, PIN_LORA_RX, PIN_LORA_TX);
}

void initGps() {
  gGps = GpsState{};

  if (PIN_GPS_RX >= 0 && PIN_GPS_TX >= 0) {
    Serial1.begin(GPS_UART_BAUD, SERIAL_8N1, PIN_GPS_RX, PIN_GPS_TX);
    gGps.rawUartEnabled = true;
    Serial.printf("[INIT] GPS UART      OK (RX:%d TX:%d @%lu)\n", PIN_GPS_RX, PIN_GPS_TX,
                  static_cast<unsigned long>(GPS_UART_BAUD));
  } else {
    Serial.println("[INIT] GPS UART      skipped (set PIN_GPS_RX/PIN_GPS_TX to enable raw GPS input)");
  }

  if (GPS_MOCK_ENABLED) {
    pushMockGpsSentence();
    Serial.println("[INIT] GPS MOCK      enabled");
  } else {
    Serial.println("[INIT] GPS MOCK      disabled");
  }
}

bool processChunkForDetection(const int16_t* chunk, const size_t sampleCount, DetectionResult& out) {
  out = DetectionResult{};

  for (size_t i = 0; i < sampleCount; ++i) {
    const float sampleMag = fabsf(static_cast<float>(chunk[i]));
    gSta = ALPHA_STA * sampleMag + (1.0f - ALPHA_STA) * gSta;
    gLta = ALPHA_LTA * sampleMag + (1.0f - ALPHA_LTA) * gLta;
    if (gLta < 1.0f) {
      gLta = 1.0f;
    }

    ++gSamplesSinceTrigger;
    const float ratio = gSta / gLta;
    if (ratio <= STA_LTA_THRESHOLD || gSamplesSinceTrigger < COOLDOWN_SAMPLES) {
      continue;
    }

    gSamplesSinceTrigger = 0;

    size_t start = (i > 128) ? (i - 128) : 0;
    size_t end = i + 896;
    if (end > sampleCount) {
      end = sampleCount;
    }
    if (end <= start) {
      start = 0;
      end = sampleCount;
    }

    out.triggerLen = end - start;
    if (out.triggerLen == 0) {
      return false;
    }
    memcpy(out.triggerRegion, chunk + start, out.triggerLen * sizeof(int16_t));

    uint16_t peakAmp = 0;
    double sumSquares = 0.0;
    for (size_t j = 0; j < out.triggerLen; ++j) {
      const int32_t v = static_cast<int32_t>(out.triggerRegion[j]);
      const uint16_t absV = static_cast<uint16_t>(abs(v));
      if (absV > peakAmp) {
        peakAmp = absV;
      }
      sumSquares += static_cast<double>(v) * static_cast<double>(v);
    }
    const float rms = sqrtf(static_cast<float>(sumSquares / static_cast<double>(out.triggerLen)));

    const float signalPower = rms * rms;
    const float noiseEstimate = (gLta / ALPHA_LTA) * ALPHA_STA;
    const float noisePower = noiseEstimate * noiseEstimate;
    const float ratioSnr = (noisePower > 1e-9f) ? (signalPower / noisePower) : 1.0f;
    float snrDb = 10.0f * log10f((ratioSnr > 1e-6f) ? ratioSnr : 1e-6f);
    snrDb = clampValue(snrDb, 0.0f, 255.0f);

    out.detected = true;
    out.peakAmplitude = peakAmp;
    out.snrDb = static_cast<uint8_t>(snrDb);
    return true;
  }

  return false;
}

void convertRawI2S32ToPcm16(const int32_t* raw, const size_t rawCount, int16_t* pcm16, size_t& outCount) {
  outCount = rawCount;
  for (size_t i = 0; i < rawCount; ++i) {
    // 24-bit signed sample is left-justified in 32-bit word.
    const int32_t s24 = raw[i] >> 8;
    int32_t s16 = s24 >> 8;
    s16 = clampValue<int32_t>(s16, -32768, 32767);
    pcm16[i] = static_cast<int16_t>(s16);
  }
}

FFTResult analyzeFFT(const int16_t* samples, const size_t length) {
  FFTResult result{};

  memset(gFFTReal, 0, sizeof(gFFTReal));
  memset(gFFTImag, 0, sizeof(gFFTImag));
  memset(gMagnitudes, 0, sizeof(gMagnitudes));

  const size_t copyLen = (length < FFT_SIZE) ? length : FFT_SIZE;
  const size_t srcOffset = (length > FFT_SIZE) ? (length - FFT_SIZE) : 0;

  for (size_t i = 0; i < copyLen; ++i) {
    gFFTReal[i] = static_cast<float>(samples[srcOffset + i]);
  }

  float mean = 0.0f;
  for (size_t i = 0; i < FFT_SIZE; ++i) {
    mean += gFFTReal[i];
  }
  mean /= static_cast<float>(FFT_SIZE);

  // Hamming window, matching simulation behavior.
  for (size_t i = 0; i < FFT_SIZE; ++i) {
    const float window = 0.54f - 0.46f * cosf((2.0f * PI * static_cast<float>(i)) / static_cast<float>(FFT_SIZE - 1));
    gFFTReal[i] = (gFFTReal[i] - mean) * window;
  }

  fftInPlace(gFFTReal, gFFTImag, FFT_SIZE);

  const size_t halfN = FFT_SIZE / 2;
  for (size_t k = 0; k <= halfN; ++k) {
    const float re = gFFTReal[k];
    const float im = gFFTImag[k];
    gMagnitudes[k] = sqrtf(re * re + im * im);
  }

  const float binWidthHz = static_cast<float>(SAMPLE_RATE) / static_cast<float>(FFT_SIZE);
  const size_t startBin = clampValue(static_cast<size_t>(WHISTLE_FREQ_MIN_HZ / binWidthHz), static_cast<size_t>(1),
                                     halfN - 1);
  const size_t endBin = clampValue(static_cast<size_t>(WHISTLE_FREQ_MAX_HZ / binWidthHz), static_cast<size_t>(1),
                                   halfN - 1);

  float bandPeakMag = 0.0f;
  size_t bandPeakBin = startBin;
  for (size_t k = startBin; k <= endBin; ++k) {
    if (gMagnitudes[k] > bandPeakMag) {
      bandPeakMag = gMagnitudes[k];
      bandPeakBin = k;
    }
  }

  float globalPeakMag = 0.0f;
  size_t globalPeakBin = 1;
  for (size_t k = 1; k <= halfN; ++k) {
    if (gMagnitudes[k] > globalPeakMag) {
      globalPeakMag = gMagnitudes[k];
      globalPeakBin = k;
    }
  }

  result.peakFreqHz = static_cast<uint16_t>(bandPeakBin * binWidthHz);
  result.globalPeakFreqHz = static_cast<uint16_t>(globalPeakBin * binWidthHz);
  result.peakMagnitude = bandPeakMag;
  result.inTargetBand = result.globalPeakFreqHz >= WHISTLE_FREQ_MIN_HZ && result.globalPeakFreqHz <= WHISTLE_FREQ_MAX_HZ;
  result.passed = (!FFT_PREFILTER_ENABLED) || result.inTargetBand;

  return result;
}

AudioMetrics computeAudioMetrics(const int16_t* samples, const size_t length) {
  AudioMetrics metrics{};
  if (length == 0) {
    return metrics;
  }

  int16_t minV = samples[0];
  int16_t maxV = samples[0];
  uint16_t peak = 0;
  double sumSquares = 0.0;
  double sum = 0.0;

  for (size_t i = 0; i < length; ++i) {
    const int32_t v = static_cast<int32_t>(samples[i]);
    if (samples[i] < minV) {
      minV = samples[i];
    }
    if (samples[i] > maxV) {
      maxV = samples[i];
    }

    const uint16_t absV = static_cast<uint16_t>(abs(v));
    if (absV > peak) {
      peak = absV;
    }
    sumSquares += static_cast<double>(v) * static_cast<double>(v);
    sum += static_cast<double>(v);
  }

  metrics.peak = peak;
  metrics.rms = sqrtf(static_cast<float>(sumSquares / static_cast<double>(length)));
  metrics.minSample = minV;
  metrics.maxSample = maxV;
  metrics.dc = static_cast<float>(sum / static_cast<double>(length));
  metrics.fingerprint = fingerprintAudio(samples, length);
  return metrics;
}

void printAudioMagnitude(const AudioMetrics& metrics) {
  const uint32_t now = millis();
  static uint32_t lastPrintMs = 0;
  if ((now - lastPrintMs) < MAG_PRINT_PERIOD_MS) {
    return;
  }
  lastPrintMs = now;

  const float ratio = gSta / ((gLta > 1.0f) ? gLta : 1.0f);
  // Serial.printf("MAG peak=%u rms=%.1f min=%d max=%d dc=%.1f sta=%.1f lta=%.1f ratio=%.2f\n",
  //               static_cast<unsigned>(metrics.peak), metrics.rms, static_cast<int>(metrics.minSample),
  //               static_cast<int>(metrics.maxSample), metrics.dc, gSta, gLta, ratio);
}

void printPerfTimings(const PerfTimings& perf) {
  static uint32_t lastPeriodicMs = 0;
  const uint32_t nowMs = millis();

  if (!perf.detected && (nowMs - lastPeriodicMs) < PERF_PRINT_PERIOD_MS) {
    return;
  }
  if (!perf.detected) {
    lastPeriodicMs = nowMs;
  }

  Serial.printf(
      "PERF us gps=%lu capture=%lu convert=%lu metrics=%lu detect=%lu fft=%lu classify=%lu rescue=%lu pkt=%lu "
      "tx=%lu total=%lu detected=%u tx_ok=%u\n",
      static_cast<unsigned long>(perf.gpsUs), static_cast<unsigned long>(perf.captureUs),
      static_cast<unsigned long>(perf.convertUs), static_cast<unsigned long>(perf.metricsUs),
      static_cast<unsigned long>(perf.detectUs), static_cast<unsigned long>(perf.fftUs),
      static_cast<unsigned long>(perf.classifyUs), static_cast<unsigned long>(perf.rescueUs),
      static_cast<unsigned long>(perf.packetBuildUs), static_cast<unsigned long>(perf.txUs),
      static_cast<unsigned long>(perf.totalUs), perf.detected ? 1U : 0U,
      (perf.txAttempted && perf.txOk) ? 1U : 0U);
}

void updateAudioFlow(const AudioMetrics& metrics) {
  ++gAudioFlow.chunksWindow;

  if (gAudioFlow.chunksWindow == 1 || metrics.fingerprint != gAudioFlow.lastFingerprint) {
    ++gAudioFlow.changedWindow;
  } else {
    ++gAudioFlow.repeatingWindow;
  }
  gAudioFlow.lastFingerprint = metrics.fingerprint;

  if (metrics.peak <= 2 && metrics.rms <= 2.0f) {
    ++gAudioFlow.quietWindow;
  }

  const uint32_t now = millis();
  if ((now - gAudioFlow.lastPrintMs) < AUDIO_FLOW_PRINT_PERIOD_MS) {
    return;
  }

  const uint32_t chunks = gAudioFlow.chunksWindow;
  const uint32_t changed = gAudioFlow.changedWindow;
  const uint32_t quiet = gAudioFlow.quietWindow;
  const uint32_t repeating = gAudioFlow.repeatingWindow;

  Serial.printf("AUDIO flow chunks=%lu changed=%lu repeating=%lu quiet=%lu capture=%s\n",
                static_cast<unsigned long>(chunks), static_cast<unsigned long>(changed),
                static_cast<unsigned long>(repeating), static_cast<unsigned long>(quiet),
                (changed > 0) ? "active" : "stuck");

  if (quiet == chunks && chunks > 0) {
    Serial.println("AUDIO hint: signal is near-zero. Check I2S mic power, WS/BCLK/SD wiring, and left/right channel.");
  }

  gAudioFlow.lastPrintMs = now;
  gAudioFlow.chunksWindow = 0;
  gAudioFlow.changedWindow = 0;
  gAudioFlow.quietWindow = 0;
  gAudioFlow.repeatingWindow = 0;
}

void computeSpectralStats(float& centroidHz, float& flatness) {
  centroidHz = 0.0f;
  flatness = 0.0f;

  const size_t halfN = FFT_SIZE / 2;
  const float binWidth = static_cast<float>(SAMPLE_RATE) / static_cast<float>(FFT_SIZE);
  float weightedSum = 0.0f;
  float magnitudeSum = 0.0f;
  float logSum = 0.0f;
  float arithMeanSum = 0.0f;
  size_t bins = 0;

  for (size_t k = 1; k <= halfN; ++k) {
    const float mag = gMagnitudes[k];
    const float freq = static_cast<float>(k) * binWidth;
    weightedSum += freq * mag;
    magnitudeSum += mag;
    logSum += logf(mag + 1e-10f);
    arithMeanSum += mag;
    ++bins;
  }

  if (magnitudeSum > 1e-9f) {
    centroidHz = weightedSum / magnitudeSum;
  }
  if (bins > 0) {
    const float geomMean = expf(logSum / static_cast<float>(bins));
    const float arithMean = arithMeanSum / static_cast<float>(bins);
    flatness = (arithMean > 1e-9f) ? (geomMean / arithMean) : 0.0f;
  }
}

ClassResult classifyWithRulesFallback(const DetectionResult& detection) {
  ClassResult result{};

  const float rms = computeRms(detection.triggerRegion, detection.triggerLen);
  const float zcr = computeZcr(detection.triggerRegion, detection.triggerLen);
  const float crest = computeCrestFactor(detection.triggerRegion, detection.triggerLen, rms);
  const float envVariance = computeEnvelopeVariance(detection.triggerRegion, detection.triggerLen, rms);

  float centroidHz = 0.0f;
  float flatness = 0.0f;
  computeSpectralStats(centroidHz, flatness);

  float confidence = 0.5f;
  uint8_t classId = CLASS_AMBIENT;

  if (rms < 30.0f) {
    classId = CLASS_AMBIENT;
    confidence = 0.90f;
  } else if (centroidHz >= 2500.0f && centroidHz <= 4000.0f && flatness < 0.1f) {
    classId = CLASS_WHISTLE;
    confidence = 0.85f;
  } else if (centroidHz >= 800.0f && centroidHz <= 4000.0f) {
    if (centroidHz > 1500.0f) {
      classId = CLASS_HUMAN_VOICE;
      confidence = 0.70f;
    } else {
      classId = CLASS_KNOCKING;
      confidence = 0.65f;
    }
  } else if (centroidHz < 300.0f) {
    if (rms > 5000.0f) {
      classId = CLASS_MACHINERY;
      confidence = 0.60f;
    } else {
      classId = CLASS_MOTOR;
      confidence = 0.60f;
    }
  } else {
    classId = CLASS_WIND;
    confidence = 0.50f;
  }

  const bool machineContext = (centroidHz < 600.0f && flatness < 0.2f);
  const float crestThreshold = machineContext ? 7.5f : 5.5f;
  const float varianceThreshold = machineContext ? 0.5f : 0.3f;

  if (rms > 5000.0f && crest > crestThreshold && envVariance > varianceThreshold) {
    classId = CLASS_IMPACT;
    confidence = 0.85f;
  } else if (rms > 15000.0f && zcr > 0.3f && !machineContext) {
    if (envVariance > 0.2f) {
      classId = CLASS_IMPACT;
      confidence = 0.70f;
    }
  }

  result.classId = classId;
  result.confidencePct = static_cast<uint8_t>(clampValue(static_cast<int>(roundf(confidence * 100.0f)), 0, 100));
  result.isTarget = isTargetClass(classId);
  if (result.isTarget) {
    result.action = PacketAction::Target;
  } else if (isLogOnlyClass(classId)) {
    result.action = PacketAction::LogOnly;
  } else {
    result.action = PacketAction::Reject;
  }

  return result;
}

ClassResult classifyWithTinyMl(const DetectionResult& detection) {
  ClassResult fallback = classifyWithRulesFallback(detection);

  if (!gTinyMelReady) {
    initTinyMlMelFilters();
  }

  if (!extractTinyMlFeatures(detection.triggerRegion, detection.triggerLen, gTinyFeatures)) {
    return fallback;
  }

  for (int i = 0; i < kTinyMlInputDim; ++i) {
    const float denom = (fabsf(kTinyMlStd[i]) < TINYML_STD_EPS) ? 1.0f : kTinyMlStd[i];
    gTinyInput[i] = (gTinyFeatures[i] - kTinyMlMean[i]) / denom;
  }

  denseLayer(gTinyInput, kTinyMlInputDim, kTinyMlW1, kTinyMlB1, kTinyMlHidden1Dim, true, gTinyHidden1);
  denseLayer(gTinyHidden1, kTinyMlHidden1Dim, kTinyMlW2, kTinyMlB2, kTinyMlHidden2Dim, true, gTinyHidden2);
  denseLayer(gTinyHidden2, kTinyMlHidden2Dim, kTinyMlW3, kTinyMlB3, kTinyMlOutputDim, false, gTinyLogits);
  softmax(gTinyLogits, kTinyMlOutputDim, gTinyProbs);

  int bestClass = 0;
  float bestProb = gTinyProbs[0];
  for (int i = 1; i < kTinyMlOutputDim; ++i) {
    if (gTinyProbs[i] > bestProb) {
      bestProb = gTinyProbs[i];
      bestClass = i;
    }
  }

  ClassResult result{};
  result.classId = static_cast<uint8_t>(clampValue(bestClass, 0, kTinyMlOutputDim - 1));
  result.confidencePct = static_cast<uint8_t>(clampValue(static_cast<int>(roundf(bestProb * 100.0f)), 0, 100));
  result.isTarget = isTargetClass(result.classId);

  if (result.isTarget) {
    result.action = PacketAction::Target;
  } else if (isLogOnlyClass(result.classId)) {
    result.action = PacketAction::LogOnly;
  } else {
    result.action = PacketAction::Reject;
  }

  return result;
}

void applyWhistleRescue(const DetectionResult& detection, const FFTResult& fftResult, ClassResult& cls) {
  if (!WHISTLE_RESCUE_ENABLED) {
    return;
  }
  if (cls.action != PacketAction::Reject) {
    return;
  }
  if (!fftResult.passed) {
    return;
  }

  const bool isRescuableClass =
      cls.classId == CLASS_ANIMAL || cls.classId == CLASS_AMBIENT || cls.classId == CLASS_WIND || cls.classId == CLASS_RAIN;
  if (!isRescuableClass) {
    return;
  }

  const int lowAmpFloor = static_cast<int>(fmaxf(120.0f, static_cast<float>(WHISTLE_RESCUE_MIN_PEAK) * 0.55f));
  const bool freqLooksWhistle = fftResult.peakFreqHz >= 2200U && fftResult.peakFreqHz <= 4200U;
  const bool ampOk = detection.peakAmplitude >= WHISTLE_RESCUE_MIN_PEAK ||
                     (detection.peakAmplitude >= static_cast<uint16_t>(lowAmpFloor) && freqLooksWhistle &&
                      cls.confidencePct < 78U);

  if (!ampOk) {
    return;
  }

  cls.classId = CLASS_WHISTLE;
  cls.confidencePct = (cls.confidencePct < 51U) ? 51U : cls.confidencePct;
  cls.isTarget = true;
  cls.action = PacketAction::Target;
}

NodePacket buildPacket(const DetectionResult& detection, const FFTResult& fftResult, const ClassResult& cls) {
  NodePacket packet{};
  packet.node_id = static_cast<uint8_t>(NODE_ID);
  packet.ts_micros = currentGpsEpochMicros();
  packet.magnitude = detection.peakAmplitude;
  packet.peak_freq_hz = fftResult.peakFreqHz;
  packet.ml_class = cls.classId;
  packet.ml_confidence = cls.confidencePct;
  packet.snr_db = detection.snrDb;
  packet.battery_pct = static_cast<uint8_t>(clampValue(static_cast<int>(roundf(gBatteryPct)), 0, 100));
  packet.temperature_c = DEFAULT_TEMP_C;
  packet.gps_hdop = currentGpsHdopField();
  packet.latitude_e7 = currentGpsLatitudeE7();
  packet.longitude_e7 = currentGpsLongitudeE7();
  packet.event_count = static_cast<uint16_t>(++gEventCount);
  packet.crc8 = 0;
  packet.reserved = 0;
  packet.crc8 = crc8(reinterpret_cast<const uint8_t*>(&packet), NODE_PACKET_CRC_LEN);
  return packet;
}

void printPacketDump(const NodePacket& packet) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&packet);
  const uint8_t expectedCrc = crc8(bytes, NODE_PACKET_CRC_LEN);
  const float lat = static_cast<float>(packet.latitude_e7) / 10000000.0f;
  const float lon = static_cast<float>(packet.longitude_e7) / 10000000.0f;

  Serial.printf(
      "TX PACKET fields: node=%u ts=%llu mag=%u f0=%u class=%u conf=%u snr=%u bat=%u temp=%d hdop=%u "
      "lat=%.7f lon=%.7f evt=%u "
      "crc=0x%02X (exp=0x%02X)\n",
      static_cast<unsigned>(packet.node_id), static_cast<unsigned long long>(packet.ts_micros),
      static_cast<unsigned>(packet.magnitude), static_cast<unsigned>(packet.peak_freq_hz),
      static_cast<unsigned>(packet.ml_class), static_cast<unsigned>(packet.ml_confidence),
      static_cast<unsigned>(packet.snr_db), static_cast<unsigned>(packet.battery_pct),
      static_cast<int>(packet.temperature_c), static_cast<unsigned>(packet.gps_hdop), lat, lon,
      static_cast<unsigned>(packet.event_count), static_cast<unsigned>(packet.crc8), static_cast<unsigned>(expectedCrc));

  Serial.print("TX PACKET raw(31B):");
  for (size_t i = 0; i < sizeof(NodePacket); ++i) {
    Serial.printf(" %02X", static_cast<unsigned>(bytes[i]));
  }
  Serial.println();
}

bool transmitPacket(const NodePacket& packet) {
  if (!waitForAuxReady(300)) {
    return false;
  }

  const size_t sent = Serial2.write(reinterpret_cast<const uint8_t*>(&packet), sizeof(packet));
  Serial2.flush();
  return sent == sizeof(packet);
}

void nodeSetup() {
  Serial.begin(115200);
  delay(500);

  Serial.println();
  Serial.println("SAR ESP32 Node starting...");
  printInitStep("BOOT", "System power-on");
  Serial.printf("[INIT] NODE_ID       %u\n", static_cast<unsigned>(NODE_ID));
  Serial.printf("[INIT] I2S PINS      BCLK:%d WS:%d SD:%d\n", PIN_I2S_BCLK, PIN_I2S_WS, PIN_I2S_SD);
  Serial.printf("[INIT] LORA PINS     TX:%d RX:%d AUX:%d M0:%d M1:%d\n", PIN_LORA_TX, PIN_LORA_RX, PIN_LORA_AUX,
                PIN_LORA_M0, PIN_LORA_M1);
  Serial.printf("[INIT] GPS PINS      TX:%d RX:%d\n", PIN_GPS_TX, PIN_GPS_RX);

  printInitStep("I2S", "Installing driver and DMA buffers");
  initI2S();
  printInitStep("I2S", "Ready");
  Serial.println("[INIT] I2S FORMAT    32-bit raw -> 16-bit PCM");
  Serial.printf("[INIT] I2S CHANNEL   %s\n", I2S_USE_RIGHT_CHANNEL ? "RIGHT" : "LEFT");

  printInitStep("LORA", "Configuring UART transport");
  initLoRaUart();
  printInitStep("LORA", "Ready");

  printInitStep("GPS", "Starting raw parser and mock feed");
  initGps();
  printInitStep("GPS", "Ready");

  printInitStep("TINYML", "Initializing neural inference runtime");
  initTinyMlMelFilters();
  Serial.printf("[INIT] TINYML DIMS   %d -> %d -> %d -> %d\n", kTinyMlInputDim, kTinyMlHidden1Dim, kTinyMlHidden2Dim,
                kTinyMlOutputDim);
  printInitStep("TINYML", "Ready");

  printInitStep("RUN", "Node ready. Streaming audio and waiting for events.");
}

void nodeLoop() {
  const int64_t loopStartUs = esp_timer_get_time();
  PerfTimings perf{};

  int64_t stageStartUs = esp_timer_get_time();
  updateGps();
  perf.gpsUs = static_cast<uint32_t>(esp_timer_get_time() - stageStartUs);

  size_t bytesRead = 0;
  stageStartUs = esp_timer_get_time();
  const esp_err_t readErr = i2s_read(I2S_PORT, gAudioRawChunk, sizeof(gAudioRawChunk), &bytesRead, portMAX_DELAY);
  perf.captureUs = static_cast<uint32_t>(esp_timer_get_time() - stageStartUs);
  if (readErr != ESP_OK) {
    perf.totalUs = static_cast<uint32_t>(esp_timer_get_time() - loopStartUs);
    printPerfTimings(perf);
    return;
  }

  const size_t rawSamplesRead = bytesRead / sizeof(int32_t);
  size_t samplesRead = 0;
  stageStartUs = esp_timer_get_time();
  convertRawI2S32ToPcm16(gAudioRawChunk, rawSamplesRead, gAudioChunk, samplesRead);
  perf.convertUs = static_cast<uint32_t>(esp_timer_get_time() - stageStartUs);
  if (samplesRead == 0) {
    perf.totalUs = static_cast<uint32_t>(esp_timer_get_time() - loopStartUs);
    printPerfTimings(perf);
    return;
  }

  stageStartUs = esp_timer_get_time();
  const AudioMetrics metrics = computeAudioMetrics(gAudioChunk, samplesRead);
  printAudioMagnitude(metrics);
  updateAudioFlow(metrics);
  perf.metricsUs = static_cast<uint32_t>(esp_timer_get_time() - stageStartUs);

  DetectionResult detection{};
  stageStartUs = esp_timer_get_time();
  if (!processChunkForDetection(gAudioChunk, samplesRead, detection)) {
    perf.detectUs = static_cast<uint32_t>(esp_timer_get_time() - stageStartUs);
    perf.totalUs = static_cast<uint32_t>(esp_timer_get_time() - loopStartUs);
    printPerfTimings(perf);
    return;
  }
  perf.detected = true;
  perf.detectUs = static_cast<uint32_t>(esp_timer_get_time() - stageStartUs);

  stageStartUs = esp_timer_get_time();
  const FFTResult fftResult = analyzeFFT(detection.triggerRegion, detection.triggerLen);
  perf.fftUs = static_cast<uint32_t>(esp_timer_get_time() - stageStartUs);

  stageStartUs = esp_timer_get_time();
  ClassResult cls = classifyWithTinyMl(detection);
  perf.classifyUs = static_cast<uint32_t>(esp_timer_get_time() - stageStartUs);

  stageStartUs = esp_timer_get_time();
  applyWhistleRescue(detection, fftResult, cls);
  perf.rescueUs = static_cast<uint32_t>(esp_timer_get_time() - stageStartUs);

  if (cls.action == PacketAction::Target && cls.isTarget) {
    stageStartUs = esp_timer_get_time();
    const NodePacket packet = buildPacket(detection, fftResult, cls);
    perf.packetBuildUs = static_cast<uint32_t>(esp_timer_get_time() - stageStartUs);

    stageStartUs = esp_timer_get_time();
    const bool txOk = transmitPacket(packet);
    perf.txUs = static_cast<uint32_t>(esp_timer_get_time() - stageStartUs);
    perf.txAttempted = true;
    perf.txOk = txOk;

    if (txOk) {
      printPacketDump(packet);
      gBatteryPct = fmaxf(0.0f, gBatteryPct - BATTERY_DRAIN_PER_EVENT);
      Serial.printf("TX evt=%u class=%s conf=%u%% peak=%u f0=%uHz snr=%u\n", static_cast<unsigned>(packet.event_count),
                    className(packet.ml_class), static_cast<unsigned>(packet.ml_confidence),
                    static_cast<unsigned>(packet.magnitude), static_cast<unsigned>(packet.peak_freq_hz),
                    static_cast<unsigned>(packet.snr_db));
    } else {
      Serial.println("LoRa TX failed (AUX timeout or UART short write)");
    }
  } else if (cls.action == PacketAction::LogOnly) {
    Serial.printf("LOG class=%s conf=%u%% peak=%u\n", className(cls.classId), static_cast<unsigned>(cls.confidencePct),
                  static_cast<unsigned>(detection.peakAmplitude));
  } else {
    Serial.printf("REJECT class=%s conf=%u%% peak=%u fft_pass=%d\n", className(cls.classId),
                  static_cast<unsigned>(cls.confidencePct), static_cast<unsigned>(detection.peakAmplitude),
                  fftResult.passed ? 1 : 0);
  }

  perf.totalUs = static_cast<uint32_t>(esp_timer_get_time() - loopStartUs);
  printPerfTimings(perf);
}

}  // namespace

void setup() { nodeSetup(); }

void loop() { nodeLoop(); }