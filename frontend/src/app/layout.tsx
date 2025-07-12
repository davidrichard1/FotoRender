import './globals.css'
import type { Metadata } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import { ThemeProvider } from '@/app/contexts/ThemeContext'
import { ErrorBoundary } from '@/components/ui/ErrorBoundary'
import { Navigation } from '@/components/ui/navigation'

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
})

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
})

export const metadata: Metadata = {
  title: 'Foto-Render - AI Image Generator',
  description:
    'High-performance local image generation using SDXL models, optimized for RTX 5090 GPU.',
  keywords:
    'ai, image generation, sdxl, diffusion, stable diffusion, gpu, rtx 5090',
  authors: [{ name: 'Foto-Render Team' }],
  robots: 'index, follow',
}

export const viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#2563eb',
  colorScheme: 'light dark',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head />
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-gray-50 dark:bg-gray-900 transition-all duration-300`}
      >
        <ErrorBoundary>
          <ThemeProvider>
            <Navigation />
            <div
              id="app"
              role="application"
              aria-label="Foto-Render AI Image Generator"
            >
              {children}
            </div>
          </ThemeProvider>
        </ErrorBoundary>
      </body>
    </html>
  )
}
