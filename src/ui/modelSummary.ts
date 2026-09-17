/**
 * How an installed model describes itself in a picker or a prompt.
 *
 * Two formats and two runtimes now coexist, and the differences between them
 * are ones a user acts on: which engine will start, whether a checkpoint will
 * execute its own Python, and whether removing it deletes their files. Those
 * belong on the row, not in a log.
 */

import type { InstalledModel } from '../domain.ts';
import { formatBytes } from '../models/modelSources.ts';
import { runtimeDisplayName, runtimeForModel } from '../worker/runtimeSession.ts';

export interface ModelSummary {
  description: string;
  detail: string;
}

export function describeModel(
  model: InstalledModel,
  options: { isDefault?: boolean } = {},
): ModelSummary {
  const runtime = runtimeForModel(model);
  return {
    description: [
      formatBytes(model.fileSize),
      runtimeDisplayName(runtime),
      ...(model.quantization ? [model.quantization] : []),
      model.source,
    ].join(' · '),
    detail: [
      model.filename,
      ...(model.managed === false ? ['in place'] : []),
      ...(model.customCodeRequired ? ['runs its own code'] : []),
      ...(options.isDefault ? ['default'] : []),
    ].join(' · '),
  };
}

/**
 * What removing this model will actually do.
 *
 * A checkpoint registered in place is the user's own directory, so removal
 * unregisters it and leaves every byte alone. Saying "delete" there would
 * describe the opposite of what happens.
 */
export function describeRemoval(model: InstalledModel): {
  message: string;
  confirmLabel: string;
} {
  if (model.managed === false) {
    return {
      message:
        `Remove ${model.name} from Local LLM? The ${formatBytes(model.fileSize)} ` +
        `checkpoint at ${model.filePath} stays exactly where it is.`,
      confirmLabel: 'Remove from List',
    };
  }
  return {
    message: `Delete ${model.name} and its ${formatBytes(model.fileSize)} of model files?`,
    confirmLabel: 'Delete Model',
  };
}

/**
 * The warning a checkpoint earns before it is registered, if any.
 *
 * Two separate concerns that happen to surface at the same moment.
 *
 * Custom code is a consent question, not an advisory: loading such a
 * checkpoint executes Python shipped inside it, with this extension's
 * privileges. Nothing downstream can make that safe, so it is asked once,
 * plainly, and recorded.
 *
 * Quantization is an advisory. Most pre-quantized checkpoints depend on
 * CUDA-only kernels; elsewhere they either fail outright or dequantize back to
 * bf16 and use more memory than the quantization saved. That is worth knowing
 * before a multi-minute load, but it is not a decision about trust.
 */
export function checkpointWarnings(checkpoint: {
  quantization?: string;
  customCodeRequired?: boolean;
}, options: { cudaAvailable?: boolean } = {}): {
  consent?: string;
  advisory?: string;
} {
  const warnings: { consent?: string; advisory?: string } = {};
  if (checkpoint.customCodeRequired) {
    warnings.consent =
      'This checkpoint ships its own Python and will execute it when loaded, ' +
      'with the same access as this extension. Register it only if you trust ' +
      'where it came from.';
  }
  // Quantized kernels are a CUDA story. Anywhere else — Metal, XPU, CPU —
  // the usual outcome is a failed load or a dequantized one. Keyed off the
  // provisioned flavor's actual accelerator, not the OS: a Windows CUDA box
  // is fine, a Linux CPU box is not.
  if (checkpoint.quantization && !options.cudaAvailable) {
    warnings.advisory =
      `This checkpoint is ${checkpoint.quantization} quantized. Those kernels are ` +
      'mostly CUDA-only, so on this computer it will either fail to load or ' +
      'dequantize to bf16 and use more memory than the quantization saved.';
  }
  return warnings;
}
