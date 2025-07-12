'use client'

import { DashboardHeader } from '@/components/dashboard'
import { ErrorBoundary } from '@/components/ui'

export function DashboardClient() {
  return (
    <ErrorBoundary
      fallback={
        <div className="p-8 text-center">
          <h2>Something went wrong</h2>
        </div>
      }
    >
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <DashboardHeader />

        <main className="container mx-auto px-4 py-2">
          <div></div>
        </main>
      </div>
    </ErrorBoundary>
  )
}
