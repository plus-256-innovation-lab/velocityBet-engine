import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import * as express from 'express';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  app.enableCors({
    origin: (origin: string | undefined, callback: (err: Error | null, allow?: boolean) => void) => {
      if (!origin || origin.includes('localhost') || origin.includes('127.0.0.1')) {
        callback(null, true);
      } else {
        const allowed = [
          'https://velocitybet-frontend-408537014080.us-central1.run.app',
          'https://velocitybet-v1-408537014080.us-central1.run.app',
        ];
        if (allowed.includes(origin)) {
          callback(null, true);
        } else {
          callback(new Error('Not allowed by CORS'));
        }
      }
    },
    credentials: true,
    methods: 'OPTIONS, POST, GET',
    allowedHeaders: 'Content-Type, Authorization',
  });

  // Enable raw body parsing specifically for the snapshot endpoint
  app.use('/api/snapshot', express.raw({ type: () => true, limit: '1000mb' }));
  // Standard JSON parsing for other endpoints
  app.use(express.json({ limit: '100mb' }));

  await app.listen(process.env.PORT ?? 3000);
}
bootstrap();
