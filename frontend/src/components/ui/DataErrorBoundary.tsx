'use client'

import React from 'react'
import { Spinner } from './spinner'

interface DataErrorFallbackProps {
  error: Error
  resetErrorBoundary: () => void
  title?: string
}

function DataErrorFallback({
  error,
  resetErrorBoundary,
  title = 'Failed to load data',
}: DataErrorFallbackProps) {
  return (
    <div className="p-4 bg-red-500/20 border border-red-500/30 rounded-lg">
      <h3 className="text-red-400 font-medium mb-2">{title}</h3>
      <p className="text-red-300 text-sm mb-3">
        {error.message || 'An unexpected error occurred'}
      </p>
      <button
        onClick={resetErrorBoundary}
        className="px-3 py-1.5 bg-red-600/80 hover:bg-red-600 text-white rounded text-sm font-medium transition-colors"
      >
        Try Again
      </button>
    </div>
  )
}

interface DataErrorBoundaryProps {
  children: React.ReactNode
  fallbackTitle?: string
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void
}

interface DataErrorBoundaryState {
  hasError: boolean
  error: Error | null
}

export class DataErrorBoundary extends React.Component<
  DataErrorBoundaryProps,
  DataErrorBoundaryState
> {
  constructor(props: DataErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): DataErrorBoundaryState {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('DataErrorBoundary caught an error:', error, errorInfo)
    this.props.onError?.(error, errorInfo)
  }

  resetErrorBoundary = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError && this.state.error) {
      return (
        <DataErrorFallback
          error={this.state.error}
          resetErrorBoundary={this.resetErrorBoundary}
          title={this.props.fallbackTitle}
        />
      )
    }

    return this.props.children
  }
}

// Loading component for Suspense fallbacks
export function DataLoadingFallback({
  message = 'Loading...',
  className = '',
}: {
  message?: string
  className?: string
}) {
  return (
    <div className={`flex items-center gap-2 p-4 text-gray-400 ${className}`}>
      <Spinner className="w-4 h-4" />
      <span className="text-sm">{message}</span>
    </div>
  )
}

// Specialized loading states for different data types
export const LoadingStates = {
  Models: () => <DataLoadingFallback message="Loading AI models..." />,
  Loras: () => <DataLoadingFallback message="Loading LoRA models..." />,
  Upscalers: () => <DataLoadingFallback message="Loading upscalers..." />,
  Embeddings: () => <DataLoadingFallback message="Loading embeddings..." />,
  Vaes: () => <DataLoadingFallback message="Loading VAE models..." />,
  Assets: () => <DataLoadingFallback message="Loading model assets..." />,
}
