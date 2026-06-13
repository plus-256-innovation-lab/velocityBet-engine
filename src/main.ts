import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import * as express from 'express';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  app.enableCors({
    origin: '*',
    methods: 'OPTIONS, POST, GET',
    allowedHeaders: 'Content-Type',
  });

  // Enable raw body parsing specifically for the snapshot endpoint
  app.use('/api/snapshot', express.raw({ type: () => true, limit: '1000mb' }));
  // Standard JSON parsing for other endpoints
  app.use(express.json({ limit: '100mb' }));

  await app.listen(process.env.PORT ?? 3000);
}
bootstrap();
