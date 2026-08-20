import * as vscode from 'vscode';

export class LocalLlmLogger implements vscode.Disposable {
  private readonly channel = vscode.window.createOutputChannel('Local LLM');

  constructor(private level: 'error' | 'info' | 'debug') {}

  setLevel(level: 'error' | 'info' | 'debug'): void {
    this.level = level;
  }

  error(message: string, error?: unknown): void {
    const detail = error instanceof Error ? `: ${error.message}` : '';
    this.write('ERROR', `${message}${detail}`);
  }

  info(message: string): void {
    if (this.level !== 'error') {
      this.write('INFO', message);
    }
  }

  debug(message: string): void {
    if (this.level === 'debug') {
      this.write('DEBUG', message);
    }
  }

  show(): void {
    this.channel.show(true);
  }

  dispose(): void {
    this.channel.dispose();
  }

  private write(kind: string, message: string): void {
    this.channel.appendLine(`[${new Date().toISOString()}] [${kind}] ${message}`);
  }
}
