import { Injectable, OnModuleInit } from '@nestjs/common';
import * as fs from 'fs';
import { resolve, join } from 'path';

@Injectable()
export class ApiService implements OnModuleInit {
  private engineModule: any;
  private storedMetadata: any = null;

  async onModuleInit() {
    // Dynamic import to load the .mjs module from the CommonJS Nest context
    // @ts-ignore
    this.engineModule = await import('../backend/engine.mjs');
    await this.engineModule.ensureInitialized();
  }

  getStatus() {
    // The path here assumes we want to check the parent directory (project root)
    // where the track snapshot is expected by the frontend.
    const snapshotPath = resolve('..', 'track-snapshot.bin');
    const metadataPath = resolve('..', 'track-metadata.json');
    const hasSnapshot = fs.existsSync(snapshotPath) && fs.existsSync(metadataPath);
    return { hasSnapshot };
  }

  setMetadata(metadata: any) {
    this.storedMetadata = metadata;
  }

  async storeSnapshot(snapshot: Buffer) {
    await this.engineModule.ensureInitialized();
    // Use the backend engine's original store method
    await this.engineModule.storeSnapshot(snapshot, this.storedMetadata);
  }

  async *streamRace(seed: any) {
    await this.engineModule.ensureInitialized();
    for await (const chunk of this.engineModule.streamRace(seed)) {
      yield JSON.stringify(chunk) + '\n';
    }
  }
}
