import { initializeAuthProxy } from '@propelauth/auth-proxy'

// Replace with your configuration
await initializeAuthProxy({
    authUrl: "https://33793539.propelauthtest.com",
    integrationApiKey: "8dbd0a58e6369c541a2fc08ff973803e5c7c50555ff123bb767e7d9d7e7c227bf5376559d7687a62806c9846b6e90be8",
    proxyPort: 8000,
    urlWhereYourProxyIsRunning: "https://pc-hospice-service-pshayzraua-ue.a.run.app",
    target: {
        host: "localhost:8000",
        protocol: 'https:'
    },
})

