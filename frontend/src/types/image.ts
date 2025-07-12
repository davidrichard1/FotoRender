export interface Image {
  id: string
  url: string
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
