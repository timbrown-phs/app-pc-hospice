import { initializeAuthProxy } from '@propelauth/auth-proxy'

// Replace with your configuration
await initializeAuthProxy({
    authUrl: "https://33793539.propelauthtest.com",
    integrationApiKey: "8dbd0a58e6369c541a2fc08ff973803e5c7c50555ff123bb767e7d9d7e7c227bf5376559d7687a62806c9846b6e90be8",
    proxyPort: 8000,
    urlWhereYourProxyIsRunning: "YOUR_PROXY_URL",
    target: {
        host: "https://app-pc-hospice-service-rzp4s3nasq-ue.a.run.app",
        protocol: 'https:'
    },
})