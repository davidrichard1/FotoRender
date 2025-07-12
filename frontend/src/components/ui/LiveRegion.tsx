'use client'

import { ReactNode } from 'react'

interface LiveRegionProps {
  children: ReactNode
  'aria-live'?: 'polite' | 'assertive' | 'off'
  'aria-label'?: string
  className?: string
}

export function LiveRegion({
  children,
  'aria-live': ariaLive = 'polite',
  'aria-label': ariaLabel,
  className = 'sr-only',
}: LiveRegionProps) {
  return (
    <div
      aria-live={ariaLive}
      aria-label={ariaLabel}
      className={className}
    >
      {children}
    </div>
  )
}
