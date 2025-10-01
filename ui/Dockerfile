# --- Build (Vite) ---
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci --no-fund --no-audit
COPY . .
RUN npm run build

# --- Serve (nginx) ---
FROM nginx:alpine

# (If you want non-root, keep the hardened version we discussed earlier; root is fine for demo.)
# RUN adduser -D -H -u 10003 web
# USER web

# main config (optional; default works too)
# COPY a minimal nginx.conf if you used the hardened non-root variant; otherwise skip.

# SPA server block + API reverse proxy
COPY <<'NGINX' /etc/nginx/conf.d/default.conf
server {
  listen 80;
  server_name _;
  root /usr/share/nginx/html;
  index index.html;

  # --- API reverse proxy (same-origin → no CORS) ---
  location /api/ {
    proxy_pass http://api:8000/;   # 'api' is the docker-compose service name
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }

  # Long-cache static assets
  location ~* \.(?:js|css|woff2?|svg|png|jpg|jpeg|gif)$ {
    add_header Cache-Control "public, max-age=31536000, immutable";
    try_files $uri =404;
  }

  # SPA fallback
  location / {
    add_header Cache-Control "no-store";
    try_files $uri /index.html;
  }
}
NGINX

COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx","-g","daemon off;"]