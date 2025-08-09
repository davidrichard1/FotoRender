export interface Image {
  id: string
  url: string
  fallbackUrl?: string  // Add fallback URL for CORS issues
  title: string
  description?: string
  originalUrl: string
  proxyUrl?: string
  platform?: string
  width?: number
  height?: number
  fileSize?: number
  mimeType?: string
  isPrivate: boolean
  isFavorite: boolean
  userId: string
  folderId?: string | null
  tags: { name: string; color?: string }[]
  createdAt: string
  updatedAt: string
  // Prompt-specific fields
  prompt?: string
  negative_prompt?: string
  isPromptData?: boolean
}

export interface Tag {
  name: string
  color: string
  videoCount: number
  id: string
}

export interface NotificationState {
  id: string
  type: 'success' | 'error' | 'info'
  message: string
  isVisible: boolean
}
