const rawBaseUrl = import.meta.env.VITE_API_URL || ''
const fallbackBaseUrl = import.meta.env.DEV ? 'http://localhost:5000' : ''

export const API_BASE_URL = (rawBaseUrl.trim() || fallbackBaseUrl).replace(/\/+$/, '')

export function buildApiUrl(path) {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return API_BASE_URL ? `${API_BASE_URL}${normalizedPath}` : normalizedPath
}
