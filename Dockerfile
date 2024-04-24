FROM node:20-slim
ENV auth_url=https://33793539.propelauthtest.com
ENV auth_api_key=8dbd0a58e6369c541a2fc08ff973803e5c7c50555ff123bb767e7d9d7e7c227bf5376559d7687a62806c9846b6e90be8
RUN mkdir -p /home/node/app/node_modules && chown -R node:node /home/node/app
WORKDIR /home/node/app
USER root
COPY package*.json ./
RUN npm install
COPY --chown=node:node . .
COPY --chown=node:node ./proxy/proxy.mjs ./   # Explicitly copy the proxy.mjs into the working directory
EXPOSE 8000
CMD [ "node", "proxy.mjs" ]

