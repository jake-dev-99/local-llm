import { Ajv } from 'ajv';
import type { ChatTool } from '../domain.js';

export type ToolDecision =
  | { kind: 'tool'; name: string; arguments: Record<string, unknown> }
  | { kind: 'final' };

const validator = new Ajv({
  allErrors: true,
  allowUnionTypes: true,
  strict: false,
  validateFormats: false,
});

export function toolDecisionResponseFormat(
  tools: readonly ChatTool[],
  toolRequired: boolean,
): Record<string, unknown> {
  return {
    type: 'json_object',
    schema: toolDecisionSchema(tools, toolRequired),
  };
}

export function parseToolDecision(
  content: string,
  tools: readonly ChatTool[],
  toolRequired: boolean,
): ToolDecision {
  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch {
    throw new Error('The schema-constrained fallback returned invalid JSON.');
  }

  if (!isRecord(parsed)) {
    throw new Error('The schema-constrained fallback returned a non-object decision.');
  }
  let validate: ReturnType<typeof validator.compile>;
  try {
    validate = validator.compile(toolDecisionSchema(tools, toolRequired));
  } catch {
    throw new Error('A supplied tool has an invalid JSON Schema.');
  }
  if (!validate(parsed)) {
    throw new Error(
      'The schema-constrained fallback returned a decision that violates the supplied tool schema.',
    );
  }
  if (
    parsed.kind === 'tool' &&
    typeof parsed.name === 'string' &&
    isRecord(parsed.arguments) &&
    tools.some((tool) => tool.function.name === parsed.name)
  ) {
    return parsed as ToolDecision;
  }
  if (!toolRequired && parsed.kind === 'final') {
    return parsed as ToolDecision;
  }
  throw new Error('The schema-constrained fallback returned an invalid decision.');
}

export function validateToolCallInput(
  name: string,
  input: unknown,
  tools: readonly ChatTool[],
): asserts input is Record<string, unknown> {
  const tool = tools.find((candidate) => candidate.function.name === name);
  if (!tool) {
    throw new Error(`The local model requested unknown tool ${name}.`);
  }
  if (!isRecord(input)) {
    throw new Error(`The local model returned non-object arguments for tool ${name}.`);
  }
  let validate: ReturnType<typeof validator.compile>;
  try {
    validate = validator.compile(tool.function.parameters ?? { type: 'object' });
  } catch {
    throw new Error(`Tool ${name} has an invalid JSON Schema.`);
  }
  if (!validate(input)) {
    throw new Error(`The local model returned input that violates the schema for tool ${name}.`);
  }
}

/**
 * The raw JSON Schema for a tool decision, for runtimes that compile the
 * grammar themselves. `toolDecisionResponseFormat` wraps this for servers
 * that take an OpenAI-style response_format envelope.
 */
export function toolDecisionJsonSchema(
  tools: readonly ChatTool[],
  toolRequired: boolean,
): Record<string, unknown> {
  return toolDecisionSchema(tools, toolRequired);
}

function toolDecisionSchema(
  tools: readonly ChatTool[],
  toolRequired: boolean,
): Record<string, unknown> {
  if (tools.length === 0) {
    throw new Error('A schema-constrained tool decision requires at least one tool.');
  }
  const names = new Set<string>();
  const toolBranches = tools.map((tool) => {
    const name = tool.function.name;
    if (!name || names.has(name)) {
      throw new Error('Tool names must be non-empty and unique.');
    }
    names.add(name);
    return {
      type: 'object',
      ...(tool.function.description ? { description: tool.function.description } : {}),
      properties: {
        kind: { type: 'string', enum: ['tool'] },
        name: { type: 'string', enum: [name] },
        arguments: tool.function.parameters ?? { type: 'object' },
      },
      required: ['kind', 'name', 'arguments'],
      additionalProperties: false,
    };
  });
  const branches: Array<Record<string, unknown>> = [...toolBranches];
  if (!toolRequired) {
    branches.push({
      type: 'object',
      properties: {
        kind: { type: 'string', enum: ['final'] },
      },
      required: ['kind'],
      additionalProperties: false,
    });
  }
  return { oneOf: branches };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
