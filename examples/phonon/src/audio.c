#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_ENGINE
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_GENERATION
#include "miniaudio.h"
#include "audio.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

struct phonon_audio {
    ma_context context;
    ma_device device;
    ma_pcm_rb queue;
    atomic_int drained;
    atomic_int finishing;
    ma_uint64 silence_frames; /* Only the callback touches this counter. */
    ma_uint64 drain_frames;
    int started;
};

/* Single consumer. No allocation, waiting, inference, or device control here. */
static void playback(ma_device *device, void *output, const void *input, ma_uint32 frames) {
    phonon_audio *audio = device->pUserData;
    float *dst = output;
    ma_uint32 copied = 0;
    int finishing = atomic_load_explicit(&audio->finishing, memory_order_acquire);
    (void)input;
    while (copied < frames) {
        ma_uint32 count = frames - copied;
        void *src;
        if (ma_pcm_rb_acquire_read(&audio->queue, &count, &src) != MA_SUCCESS || count == 0) break;
        memcpy(dst + copied, src, count * sizeof(float));
        ma_pcm_rb_commit_read(&audio->queue, count);
        copied += count;
    }
    memset(dst + copied, 0, (frames - copied) * sizeof(float));
    if (copied) audio->silence_frames = 0;
    /* Continue sending silence after EOF so the final samples pass through the
       resampler and device buffers before the host stops the device. */
    if (finishing && copied < frames) {
        ma_uint64 tail = audio->drain_frames;
        if (audio->silence_frames < tail) {
            audio->silence_frames += frames - copied;
            if (audio->silence_frames >= tail) atomic_store_explicit(&audio->drained, 1, memory_order_release);
        }
    }
}

int phonon_audio_create(uint32_t sample_rate, int null_backend, phonon_audio **out) {
    phonon_audio *audio = calloc(1, sizeof(*audio));
    ma_result result;
    ma_backend backend = ma_backend_null;
    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    *out = NULL;
    if (sample_rate < 8000 || sample_rate > 384000) { free(audio); return MA_INVALID_ARGS; }
    if (!audio) return MA_OUT_OF_MEMORY;
    atomic_init(&audio->finishing, 0);
    atomic_init(&audio->drained, 0);
    result = ma_context_init(null_backend ? &backend : NULL, null_backend ? 1 : 0, NULL, &audio->context);
    if (result != MA_SUCCESS) goto fail_context;
    /* Never silently fall back to the null device in normal playback. */
    if (!null_backend && audio->context.backend == ma_backend_null) { result = MA_NO_BACKEND; goto fail_queue; }
    result = ma_pcm_rb_init(ma_format_f32, 1, sample_rate, NULL, NULL, &audio->queue);
    if (result != MA_SUCCESS) goto fail_queue;
    config.playback.format = ma_format_f32;
    config.playback.channels = 1;
    config.sampleRate = sample_rate;
    config.periodSizeInMilliseconds = 10;
    config.dataCallback = playback;
    config.pUserData = audio;
    result = ma_device_init(&audio->context, &config, &audio->device);
    if (result != MA_SUCCESS) goto fail_device;
    /* Account for the backend's negotiated buffer size, plus 200 ms for the
       resampler and downstream audio server. This is a drain margin, not an
       exact hardware playback-position measurement. */
    audio->drain_frames = sample_rate / 5;
    if (audio->device.playback.internalSampleRate) {
        audio->drain_frames += (ma_uint64)audio->device.playback.internalPeriodSizeInFrames *
            audio->device.playback.internalPeriods * sample_rate / audio->device.playback.internalSampleRate;
    }
    *out = audio;
    return MA_SUCCESS;
fail_device:
    ma_pcm_rb_uninit(&audio->queue);
fail_queue:
    ma_context_uninit(&audio->context);
fail_context:
    free(audio);
    return result;
}

/* Single producer; backpressure happens on the generation thread only. */
int phonon_audio_write(phonon_audio *audio, const float *samples, size_t count) {
    if (atomic_load_explicit(&audio->finishing, memory_order_acquire)) return MA_INVALID_OPERATION;
    while (count) {
        ma_uint32 n = (ma_uint32)(count > 4096 ? 4096 : count);
        void *dst;
        ma_result result = ma_pcm_rb_acquire_write(&audio->queue, &n, &dst);
        if (result != MA_SUCCESS) return result;
        if (n == 0) {
            if (!ma_device_is_started(&audio->device)) return MA_DEVICE_NOT_STARTED;
            ma_sleep(1);
            continue;
        }
        memcpy(dst, samples, n * sizeof(float));
        ma_pcm_rb_commit_write(&audio->queue, n);
        samples += n;
        count -= n;
        if (!audio->started) {
            result = ma_device_start(&audio->device);
            if (result != MA_SUCCESS) return result;
            audio->started = 1;
        }
    }
    return MA_SUCCESS;
}

int phonon_audio_finish(phonon_audio *audio) {
    if (atomic_exchange_explicit(&audio->finishing, 1, memory_order_acq_rel)) return MA_INVALID_OPERATION;
    if (!audio->started) return MA_SUCCESS;
    if (!ma_device_is_started(&audio->device)) return MA_DEVICE_NOT_STARTED;
    while (!atomic_load_explicit(&audio->drained, memory_order_acquire)) {
        if (!ma_device_is_started(&audio->device)) return MA_DEVICE_NOT_STARTED;
        ma_sleep(1);
    }
    return ma_device_stop(&audio->device);
}

void phonon_audio_destroy(phonon_audio *audio) {
    if (!audio) return;
    ma_device_uninit(&audio->device);
    ma_pcm_rb_uninit(&audio->queue);
    ma_context_uninit(&audio->context);
    free(audio);
}
