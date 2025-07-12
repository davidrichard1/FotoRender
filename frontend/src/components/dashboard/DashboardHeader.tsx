'use client'

import React from 'react'
import { Button, ThemeToggle } from '@/components/ui'

export const DashboardHeader = () => (
  <header
    className="bg-white/80 dark:bg-gray-800/80 border-b border-gray-200 dark:border-gray-700 p-4 sm:p-6 transition-all duration-300 backdrop-blur-sm relative z-10"
    role="banner"
  >
    <nav
      className="max-w-[1400px] mx-auto flex items-center justify-between"
      role="navigation"
      aria-label="Main navigation"
      id="navigation"
    >
      <div className="flex items-center gap-3">
        <div className="flex flex-col items-center group cursor-pointer">
          <span className="text-sm font-bold bg-gradient-to-r from-amber-600 to-yellow-500 bg-clip-text text-transparent leading-none">
              Foto Render AI
          </span>
        </div>
        <div
          className="h-8 w-px bg-gradient-to-b from-amber-400/70 to-yellow-500/70"
          aria-hidden="true"
        ></div>
        <h1 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white">
          <span className="bg-gradient-to-r from-gray-900 to-gray-700 dark:from-white dark:to-gray-200 bg-clip-text text-transparent">
              Image
          </span>{' '}
          <span className="bg-gradient-to-r from-amber-600 to-yellow-500 bg-clip-text text-transparent">
              Generation
          </span>
        </h1>
      </div>

      <div className="flex items-center gap-2 sm:gap-4">
        {/* Navigation Links */}
        <div className="hidden sm:flex items-center gap-2 mr-4">
          <Button
            variant="ghost"
            className="text-sm hover:bg-blue-50 hover:text-blue-700 dark:hover:bg-blue-900/20 dark:hover:text-blue-300 transition-all duration-200"
            onClick={() => (window.location.href = '/dashboard')}
            aria-label="Go to dashboard"
          >
              📊 Dashboard
          </Button>
          <Button
            variant="ghost"
            className="text-sm hover:bg-purple-50 hover:text-purple-700 dark:hover:bg-purple-900/20 dark:hover:text-purple-300 transition-all duration-200"
            onClick={() => (window.location.href = '/image-generation')}
            aria-label="Go to AI image generation"
          >
              🎨 AI Generation
          </Button>
          <Button
            variant="ghost"
            className="text-sm hover:bg-green-50 hover:text-green-700 dark:hover:bg-green-900/20 dark:hover:text-green-300 transition-all duration-200"
            onClick={() => (window.location.href = '/prompts')}
            aria-label="Go to prompt library"
          >
              💡 Prompts
          </Button>
          <Button
            variant="ghost"
            className="text-sm hover:bg-red-50 hover:text-red-700 dark:hover:bg-red-900/20 dark:hover:text-red-300 transition-all duration-200"
            onClick={() => (window.location.href = '/admin')}
            aria-label="Go to admin panel"
          >
              ⚙️ Admin
          </Button>
        </div>

        <span className="hidden sm:inline text-gray-500 dark:text-dark-tertiary text-sm">
            Welcome, Guest!
        </span>
        <ThemeToggle />
      </div>
    </nav>
  </header>
)
