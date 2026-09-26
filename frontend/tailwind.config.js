/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          primary: '#F4F7FA',
          secondary: '#EAF1F6',
        },
        card: {
          DEFAULT: '#FFFFFF',
          border: '#DCE5EC',
        },
        text: {
          primary: '#102A43',
          secondary: '#64748B',
          muted: '#94A3B8',
        },
        brand: {
          navy: '#123B6D',
          blue: '#1769AA',
          light: '#E8F2FA',
        },
        status: {
          success: '#26966F',
          warning: '#D99A28',
          danger: '#D94B55',
        }
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      boxShadow: {
        card: '0 2px 8px rgba(16, 42, 67, 0.05)',
      }
    },
  },
  plugins: [],
}
