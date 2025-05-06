# Icons for TehqeeqCast

This directory contains icons used in the TehqeeqCast application.

## Icon Files

- `markov-icon.png`: Icon for the Markov Chain model
- `hmm-icon.png`: Icon for the Hidden Markov Model
- `queue-icon.png`: Icon for the M/M/1 Queue model

## Usage

These icons are used in the home page to represent each model. If the icons are not available, the application will fall back to using emoji characters.

## Adding New Icons

To add new icons:
1. Place the icon file in this directory
2. Use the following format in your components:
```jsx
<img 
  src="/icons/your-icon.png" 
  alt="Description" 
  width="48" 
  height="48"
  onError={(e) => { 
    e.currentTarget.src = '🔄'; // Fallback emoji
    e.currentTarget.style.fontSize = '3rem'; 
  }} 
/>
```
