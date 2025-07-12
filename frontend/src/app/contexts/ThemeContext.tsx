'use client'

import React, {
  createContext, useContext, useState, useEffect,
} from 'react'

type Theme = 'light' | 'dark'

interface ThemeContextType {
  theme: Theme
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
  isLoading: boolean
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

// Helper functions for local storage
const getLocalTheme = (): Theme => {
  if (typeof window === 'undefined') return 'dark'

  const stored = localStorage.getItem('foto-render-theme')
  if (stored === 'dark' || stored === 'light') return stored

  // Default to dark mode (user preference)
  return 'dark'
}

const setLocalTheme = (theme: Theme) => {
  if (typeof window === 'undefined') return
  localStorage.setItem('foto-render-theme', theme)
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<Theme>('dark')
  const [isLoading, setIsLoading] = useState(true)

  // Load theme from local storage on mount
  useEffect(() => {
    const localTheme = getLocalTheme()
    setThemeState(localTheme)
    document.documentElement.classList.toggle('dark', localTheme === 'dark')
    setIsLoading(false)
  }, [])

  // Update DOM when theme changes
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [theme])

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme)
    setLocalTheme(newTheme)
  }

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    setTheme(newTheme)
  }

  return (
    <ThemeContext.Provider value={{
      theme, setTheme, toggleTheme, isLoading,
    }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}
