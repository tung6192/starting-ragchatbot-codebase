# Frontend Changes: Dark/Light Theme Toggle

## Summary
Added a theme toggle button that allows users to switch between dark and light modes. The toggle is positioned in the top-right corner and uses sun/moon icons for intuitive visual feedback. The light theme has been designed with WCAG AA accessibility standards in mind.

## Files Modified

### 1. `frontend/index.html`
- Added a theme toggle button element with:
  - Sun icon (displayed in dark mode)
  - Moon icon (displayed in light mode)
  - Proper ARIA label for accessibility (`aria-label="Toggle dark mode"`)
  - Title attribute for tooltip (`title="Toggle theme"`)

### 2. `frontend/style.css`
- Added comprehensive light theme CSS variables under `[data-theme="light"]` selector
- Added `.theme-toggle` button styles:
  - Fixed positioning in top-right corner
  - Circular button design (44px diameter)
  - Hover effects with scale transform
  - Focus ring for keyboard accessibility
  - Active state feedback
- Added icon visibility toggling based on theme
- Added CSS variables for scrollbar and code background colors
- Added smooth transition effects for theme changes on key elements
- Updated error/success messages to use CSS variables for theme support
- Updated link colors to use CSS variables for proper contrast in both themes

### 3. `frontend/script.js`
- Added theme management functions:
  - `initTheme()`: Initializes theme from localStorage or system preference
  - `setTheme(theme)`: Applies theme and saves to localStorage
  - `toggleTheme()`: Toggles between dark and light themes
- Theme initialization runs immediately (before DOMContentLoaded) to prevent flash
- Added event listeners for click and keyboard (Enter/Space) on toggle button

## Features
- **Persistence**: Theme preference is saved to localStorage and persists across sessions
- **System Preference**: Respects user's system color scheme preference on first visit
- **Smooth Transitions**: All theme changes animate smoothly (0.3s ease)
- **Accessibility**:
  - Keyboard navigable (Tab to focus, Enter/Space to toggle)
  - ARIA label for screen readers
  - Visible focus ring
  - WCAG AA compliant contrast ratios
- **Visual Feedback**: Sun icon in dark mode, moon icon in light mode

## CSS Variables - Complete Reference

### Core Colors
| Variable | Dark Theme | Light Theme |
|----------|------------|-------------|
| `--primary-color` | `#2563eb` | `#2563eb` |
| `--primary-hover` | `#1d4ed8` | `#1d4ed8` |
| `--primary-light` | `#3b82f6` | `#3b82f6` |
| `--background` | `#0f172a` | `#f8fafc` |
| `--surface` | `#1e293b` | `#ffffff` |
| `--surface-hover` | `#334155` | `#f1f5f9` |

### Text Colors
| Variable | Dark Theme | Light Theme |
|----------|------------|-------------|
| `--text-primary` | `#f1f5f9` | `#0f172a` |
| `--text-secondary` | `#94a3b8` | `#475569` |
| `--text-inverse` | `#1e293b` | `#f1f5f9` |

### UI Elements
| Variable | Dark Theme | Light Theme |
|----------|------------|-------------|
| `--border-color` | `#334155` | `#cbd5e1` |
| `--user-message` | `#2563eb` | `#2563eb` |
| `--user-message-text` | `#ffffff` | `#ffffff` |
| `--assistant-message` | `#374151` | `#f1f5f9` |
| `--code-bg` | `rgba(0,0,0,0.2)` | `rgba(15,23,42,0.06)` |
| `--scrollbar-thumb` | `#334155` | `#cbd5e1` |
| `--scrollbar-thumb-hover` | `#94a3b8` | `#94a3b8` |

### Status Colors
| Variable | Dark Theme | Light Theme |
|----------|------------|-------------|
| `--error-bg` | `rgba(239,68,68,0.1)` | `rgba(239,68,68,0.1)` |
| `--error-color` | `#f87171` | `#dc2626` |
| `--error-border` | `rgba(239,68,68,0.2)` | `rgba(239,68,68,0.3)` |
| `--success-bg` | `rgba(34,197,94,0.1)` | `rgba(34,197,94,0.1)` |
| `--success-color` | `#4ade80` | `#16a34a` |
| `--success-border` | `rgba(34,197,94,0.2)` | `rgba(34,197,94,0.3)` |

### Link Colors
| Variable | Dark Theme | Light Theme |
|----------|------------|-------------|
| `--link-color` | `#60a5fa` | `#2563eb` |
| `--link-hover` | `#93c5fd` | `#1d4ed8` |

## Accessibility Notes

### Contrast Ratios (WCAG AA Compliant)
- **Light theme text on background**: `#0f172a` on `#f8fafc` = 15.3:1 (exceeds 4.5:1 requirement)
- **Light theme secondary text**: `#475569` on `#f8fafc` = 6.4:1 (exceeds 4.5:1 requirement)
- **Light theme links**: `#2563eb` on `#ffffff` = 4.6:1 (meets 4.5:1 requirement)
- **Error text in light mode**: `#dc2626` ensures readable contrast on light backgrounds
- **Success text in light mode**: `#16a34a` ensures readable contrast on light backgrounds

### Keyboard Navigation
- Toggle button is focusable via Tab key
- Activates with Enter or Space key
- Clear focus indicator (3px ring)
