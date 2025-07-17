import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  /* config options here */
  experimental: {
    optimizePackageImports: ['@pandacss/dev'],
  },
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
    // Image optimization settings for better caching
    minimumCacheTTL: 86400, // 24 hours in seconds
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
  // Headers for better caching
  async headers() {
    return [
      {
        source: '/_next/image/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=86400, s-maxage=31536000', // 24h for browser, 1 year for CDN
          },
        ],
      },
      {
        source: '/images/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=86400, immutable',
          },
        ],
      },
    ]
  },
}

export default nextConfig
