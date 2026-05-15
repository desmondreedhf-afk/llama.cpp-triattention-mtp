#pragma once

#include "../../include/ggml.h"

// TurboQuant InnerQ per-channel equalization — cross-TU shared state
// The host-side state lives in turbo-innerq.cu; device-side state is per-TU
// in turbo-quant.cuh (only set-rows.cu needs device access).

#define INNERQ_MAX_CHANNELS 128

// Host-side shared state (defined in turbo-innerq.cu)
// Wrapped in extern "C" to match C-linkage declarations in llama-kv-cache.cpp
#ifdef __cplusplus
extern "C" {
#endif
GGML_API bool  g_innerq_finalized;
GGML_API float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS];
GGML_API bool turbo_innerq_needs_tensor_update(void);
GGML_API void turbo_innerq_mark_tensor_updated(void);
#ifdef __cplusplus
}
#endif

// Called from set-rows.cu after InnerQ finalization to publish scale_inv
void turbo_innerq_publish(const float * scale_inv, int group_size);
