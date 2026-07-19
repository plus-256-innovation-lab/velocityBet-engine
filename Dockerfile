# Stage 1: Build the application
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /usr/src/app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install all dependencies (including devDependencies)
RUN npm ci

# Copy the rest of the application source code
COPY . .

# Build the NestJS application
RUN npm run build

# Stage 2: Setup production environment
FROM node:20-alpine AS production

# Set NODE_ENV to production
ENV NODE_ENV=production

# Set working directory
WORKDIR /usr/src/app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install only production dependencies
RUN npm ci --only=production

# Copy built files from the builder stage
COPY --from=builder /usr/src/app/dist ./dist

# Copy the xgboost JSON models + meta required at runtime
COPY xgboost_model_winner.json ./
COPY xgboost_model_winner_*.json ./
COPY model_meta.json ./
COPY track-snapshot.bin.gz track-metadata.json ./
COPY track-centerline.json ./

# Expose the default Cloud Run port (8080)
EXPOSE 8080

# Cloud Run sets the PORT environment variable to 8080, NestJS should listen on it.
CMD ["node", "dist/main"]
