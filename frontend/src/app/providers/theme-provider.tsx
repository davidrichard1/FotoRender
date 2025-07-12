'use client'

import React, {
  createContext, useContext, useEffect, useState,
} from 'react'

type Theme = 'light' | 'dark' | 'system'

type ThemeProviderContextType = {
  theme: Theme
  setTheme: (theme: Theme) => void
  actualTheme: 'light' | 'dark' // The resolved theme (system preference resolved)
}

const ThemeProviderContext = createContext<
  ThemeProviderContextType | undefined
>(undefined)

export function useTheme() {
  const context = useContext(ThemeProviderContext)
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}

interface ThemeProviderProps {
  children: React.ReactNode
  defaultTheme?: Theme
  storageKey?: string
}

export function ThemeProvider({
  children,
  defaultTheme = 'system',
  storageKey = 'theme',
}: ThemeProviderProps) {
  const [theme, setTheme] = useState<Theme>(defaultTheme)
  const [actualTheme, setActualTheme] = useState<'light' | 'dark'>('light')

  useEffect(() => {
    // Load theme from localStorage on mount
    try {
      const storedTheme = localStorage.getItem(storageKey) as Theme
      if (storedTheme && ['light', 'dark', 'system'].includes(storedTheme)) {
        setTheme(storedTheme)
      }
    } catch (error) {
      // Fallback to default theme if localStorage is not available
      console.warn('Failed to load theme from localStorage:', error)
    }
  }, [storageKey])

  useEffect(() => {
    // Function to get system preference
    const getSystemTheme = (): 'light' | 'dark' => {
      if (typeof window !== 'undefined' && window.matchMedia) {
        return window.matchMedia('(prefers-color-scheme: dark)').matches
          ? 'dark'
          : 'light'
      }
      return 'light'
    }

    // Resolve actual theme
    let resolvedTheme: 'light' | 'dark'
    if (theme === 'system') {
      resolvedTheme = getSystemTheme()
    } else {
      resolvedTheme = theme
    }

    setActualTheme(resolvedTheme)

    // Apply theme to document
    const root = window.document.documentElement
    root.setAttribute('data-color-mode', resolvedTheme)

    // Store theme in localStorage
    try {
      localStorage.setItem(storageKey, theme)
    } catch (error) {
      console.warn('Failed to save theme to localStorage:', error)
    }

    // Listen for system theme changes when theme is 'system'
    if (theme === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
      const handleChange = (e: MediaQueryListEvent) => {
        setActualTheme(e.matches ? 'dark' : 'light')
        root.setAttribute('data-color-mode', e.matches ? 'dark' : 'light')
      }

      mediaQuery.addEventListener('change', handleChange)
      return () => mediaQuery.removeEventListener('change', handleChange)
    }
  }, [theme, storageKey])

  const value = {
    theme,
    setTheme,
    actualTheme,
  }

  return (
    <ThemeProviderContext.Provider value={value}>
      {children}
    </ThemeProviderContext.Provider>
  )
}
