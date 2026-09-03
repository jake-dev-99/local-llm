export interface ModelLoadProgressMessages {
  loading: string;
  loaded: string;
}

export function modelLoadProgress(options: {
  modelResident: boolean;
  toolCallsPossible: boolean;
  agentRequest: boolean;
}): ModelLoadProgressMessages | undefined {
  if (options.modelResident || options.toolCallsPossible || options.agentRequest) {
    return undefined;
  }
  return {
    loading: 'Loading Model into Memory',
    loaded: 'Model loaded successfully - Response Processing',
  };
}
