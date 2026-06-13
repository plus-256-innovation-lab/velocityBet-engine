import { Controller, Get, Post, Body, Req, Res, HttpStatus } from '@nestjs/common';
import { ApiService } from './api.service';
import type { Request, Response } from 'express';

@Controller('api')
export class ApiController {
  constructor(private readonly apiService: ApiService) {}

  @Get('status')
  getStatus() {
    return this.apiService.getStatus();
  }

  @Post('metadata')
  setMetadata(@Body() metadata: any, @Res() res: Response) {
    this.apiService.setMetadata(metadata);
    res.status(HttpStatus.OK).send();
  }

  @Post('snapshot')
  async storeSnapshot(@Req() req: Request, @Res() res: Response) {
    try {
      console.log(`[API] Received POST /api/snapshot. Body length: ${req.body?.length}`);
      await this.apiService.storeSnapshot(req.body);
      console.log(`[API] Snapshot stored successfully.`);
      res.status(HttpStatus.OK).json({ status: 'ok' });
    } catch (err: any) {
      console.error('[API] Error in /api/snapshot:', err);
      res.status(HttpStatus.INTERNAL_SERVER_ERROR).json({ error: err.message });
    }
  }

  @Post('precompute')
  async precompute(@Body() body: any, @Res() res: Response) {
    try {
      console.log(`[API] Received POST /api/precompute`);
      const { seed } = body;
      
      res.status(HttpStatus.OK).set({
        'Content-Type': 'application/json',
        'Transfer-Encoding': 'chunked'
      });

      let frameCount = 0;
      for await (const chunk of this.apiService.streamRace(seed)) {
        res.write(chunk);
        frameCount++;
        if (frameCount % 1000 === 0) console.log(`[API] Streamed ${frameCount} frames...`);
      }
      console.log(`[API] Stream completed. Total frames: ${frameCount}`);
      res.end();
    } catch (err: any) {
      console.error('[API] Error in /api/precompute:', err);
      if (!res.headersSent) {
        res.status(HttpStatus.INTERNAL_SERVER_ERROR).json({ error: err.message });
      } else {
        res.end();
      }
    }
  }
}
