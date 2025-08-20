# MCP Stack Mobile Applications - 40by6

Cross-platform mobile applications for iOS and Android providing full access to the MCP Stack.

## Architecture

We use React Native with Expo for cross-platform development, providing:
- Single codebase for iOS and Android
- Native performance and UI
- Access to device features
- Over-the-air updates
- Push notifications

## Features

### Core Features
- **Authentication**: Biometric login, OAuth2, API keys
- **Dashboard**: Real-time metrics and monitoring
- **Scraper Management**: Monitor and control scrapers
- **Data Visualization**: Interactive charts and graphs
- **Alerts**: Push notifications for critical events
- **Offline Mode**: Local caching and sync
- **Security**: End-to-end encryption, secure storage

### Advanced Features
- **AR Visualization**: Augmented reality data views
- **Voice Control**: Voice commands and queries
- **Gesture Controls**: Swipe actions and shortcuts
- **Widgets**: Home screen widgets for quick access
- **Apple Watch / Wear OS**: Companion apps
- **Dark Mode**: Full theme support
- **Accessibility**: VoiceOver/TalkBack support

## Project Structure

```
mobile/
├── app/                    # Expo app directory
│   ├── (tabs)/            # Tab navigation screens
│   ├── (auth)/            # Authentication screens
│   ├── components/        # Reusable components
│   ├── hooks/            # Custom React hooks
│   ├── services/         # API and service layers
│   ├── stores/           # State management
│   ├── utils/            # Utility functions
│   └── constants/        # App constants
├── assets/               # Images, fonts, etc.
├── ios/                  # iOS-specific code
├── android/              # Android-specific code
└── web/                  # Web build (optional)
```

## Getting Started

### Prerequisites
- Node.js 16+
- Expo CLI
- iOS Simulator (Mac only) or Android Studio
- Physical device for testing (recommended)

### Installation

```bash
cd mobile
npm install
```

### Development

```bash
# Start Expo development server
npm start

# Run on iOS
npm run ios

# Run on Android
npm run android

# Run on web (if enabled)
npm run web
```

### Building

```bash
# Build for iOS
eas build --platform ios

# Build for Android
eas build --platform android

# Build for both
eas build --platform all
```

## Configuration

### Environment Variables

Create `.env` file:
```
API_URL=https://api.openpolicy.me
SENTRY_DSN=your-sentry-dsn
ONESIGNAL_APP_ID=your-onesignal-id
```

### App Configuration

Edit `app.json`:
```json
{
  "expo": {
    "name": "MCP Stack",
    "slug": "mcp-stack",
    "version": "1.0.0",
    "orientation": "portrait",
    "icon": "./assets/icon.png",
    "splash": {
      "image": "./assets/splash.png",
      "resizeMode": "contain",
      "backgroundColor": "#0a1929"
    }
  }
}
```

## Testing

```bash
# Run unit tests
npm test

# Run E2E tests
npm run test:e2e

# Run on device
expo start --tunnel
```

## Deployment

### App Store (iOS)

1. Configure certificates in Apple Developer Console
2. Build with EAS: `eas build --platform ios`
3. Submit with EAS: `eas submit --platform ios`

### Google Play (Android)

1. Configure signing in Google Play Console
2. Build with EAS: `eas build --platform android`
3. Submit with EAS: `eas submit --platform android`

## Security

- All API communications use HTTPS
- Sensitive data stored in secure storage (Keychain/Keystore)
- Biometric authentication for app access
- Certificate pinning for API requests
- Obfuscation and minification in production builds

## Performance

- Lazy loading of screens and components
- Image optimization and caching
- Background task management
- Memory leak prevention
- Crash reporting with Sentry

## Accessibility

- Full VoiceOver (iOS) and TalkBack (Android) support
- Dynamic font sizing
- High contrast mode
- Reduced motion options
- Screen reader optimized navigation

## License

MIT License - see LICENSE file for details.