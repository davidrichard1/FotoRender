import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  /* config options here */
  experimental: {
    optimizePackageImports: ['@pandacss/dev'],
  },
  // Allow cross-origin requests from local network IPs during development
  allowedDevOrigins: [
    '10.0.0.88',
    '192.168.1.0/24', // Common local network range
    '10.0.0.0/24',    // Your network range
    '172.16.0.0/24',  // Docker network range
  ],
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'foto.mylinkbuddy.com',
        port: '',
        pathname: '/**',
      },
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
        pathname: '/**',
      },
    ],
    // Image optimization settings for better caching (less aggressive for network access)
    minimumCacheTTL: 3600, // 1 hour in seconds (reduced from 24 hours)
    formats: ['image/avif', 'image/webp'], // Modern formats for better compression
    dangerouslyAllowSVG: false,
    contentSecurityPolicy: "default-src 'self'; script-src 'none'; sandbox;",
    // Quality settings for different sizes
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    // Disable optimization for external images to avoid CORS issues
    unoptimized: false,
    loader: 'default',
  },
  // Headers for better caching + CORS for iPad access
  async headers() {
    return [
      {
        // Allow all API routes to be accessed from iPad
        source: '/api/:path*',
        headers: [
          {
            key: 'Access-Control-Allow-Origin',
            value: '*', // Allow all origins for API routes
          },
          {
            key: 'Access-Control-Allow-Methods',
            value: 'GET, POST, PUT, DELETE, OPTIONS',
          },
          {
            key: 'Access-Control-Allow-Headers',
            value: 'Content-Type, Authorization',
          },
        ],
      },
      {
        // Allow all pages to be accessed from iPad  
        source: '/:path*',
        headers: [
          {
            key: 'Access-Control-Allow-Origin',
            value: '*',
          },
        ],
      },
      {
        source: '/_next/image/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=3600, s-maxage=86400', // 1h for browser, 24h for CDN (less aggressive)
          },
          {
            key: 'Access-Control-Allow-Origin',
            value: '*',
          },
        ],
      },
      {
        source: '/images/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=3600', // 1 hour, removed immutable for network flexibility
          },
          {
            key: 'Access-Control-Allow-Origin',
            value: '*',
          },
        ],
      },
    ]
  },
}

export default nextConfig
