'use client'

import React from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

export function Navigation() {
  const pathname = usePathname()

  const navItems = [
    { href: '/', label: '🎨 Generate' },
    { href: '/dashboard', label: '📊 Dashboard' },
    { href: '/admin', label: '⚙️ Admin' },
  ]

  return (
    <header className="sticky top-0 z-50 w-full border-b border-[#48484A] bg-[#2C2C2E]/95 backdrop-blur shadow-lg">
      <div className="container mx-auto px-4 flex h-14 items-center">
        <div className="mr-6 flex items-center">
          <Link className="flex items-center space-x-2" href="/">
            <div className="w-6 h-6 bg-gradient-to-r from-[#00D4AA] to-[#FF6B35] rounded-md flex items-center justify-center">
              <span className="text-white font-bold text-sm">F</span>
            </div>
            <span className="font-bold text-lg text-white">Foto-Render</span>
          </Link>
        </div>

        <nav className="flex items-center space-x-6">
          {navItems.map((item) => {
            const isActive = pathname === item.href
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? 'bg-[#00D4AA]/20 text-[#00D4AA] border border-[#00D4AA]/30 shadow-lg shadow-[#00D4AA]/20'
                    : 'text-gray-300 hover:text-white hover:bg-[#48484A] border border-transparent'
                }`}
              >
                {item.label}
              </Link>
            )
          })}
        </nav>

        <div className="flex flex-1 items-center justify-end">
          <div className="text-sm text-[#00D4AA]">
            AI Image Generation Studio
          </div>
        </div>
      </div>
    </header>
  )
}
