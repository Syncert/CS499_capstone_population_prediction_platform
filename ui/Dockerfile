# --- Dev image (Vite server) ---
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm npm install --no-fund --no-audit
COPY . .
EXPOSE 5173
CMD ["npm","run","dev","--","--host","0.0.0.0"]

# --- Build + static serve (nginx) ---
# FROM node:20-alpine AS build
# WORKDIR /app
# COPY package*.json ./
# RUN npm ci
# COPY . .
# RUN npm run build
#
# FROM nginx:alpine
# COPY --from=build /app/dist /usr/share/nginx/html
# EXPOSE 80
# CMD ["nginx","-g","daemon off;"]