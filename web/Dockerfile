FROM node:20-alpine

WORKDIR /app

# Install dependencies first for better caching
COPY package*.json ./
RUN npm ci --no-audit --no-fund

# Copy the rest of the app
COPY . .

EXPOSE 5173

# Use compose command override to run dev; keep a sensible default here
CMD ["npm", "run", "dev", "--", "--host"]


