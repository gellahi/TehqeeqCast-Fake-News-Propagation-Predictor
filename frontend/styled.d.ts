import 'styled-components';

declare module 'styled-components' {
  export interface DefaultTheme {
    colors: {
      background: string;
      surface: string;
      surfaceLight: string;
      primary: string;
      secondary: string;
      accent: string;
      textPrimary: string;
      textSecondary: string;
      error: string;
    };
    shadows: {
      small: string;
      medium: string;
      large: string;
      glow: (color: string) => string;
    };
    transitions: {
      fast: string;
      medium: string;
      slow: string;
    };
    borderRadius: {
      small: string;
      medium: string;
      large: string;
      round: string;
    };
    spacing: {
      xs: string;
      sm: string;
      md: string;
      lg: string;
      xl: string;
      xxl: string;
    };
    breakpoints: {
      sm: string;
      md: string;
      lg: string;
      xl: string;
    };
    typography: {
      fontFamily: string;
      fontWeights: {
        light: number;
        regular: number;
        medium: number;
        semibold: number;
        bold: number;
      };
      sizes: {
        xs: string;
        sm: string;
        md: string;
        lg: string;
        xl: string;
        xxl: string;
        h1: string;
        h2: string;
        h3: string;
        h4: string;
      };
      lineHeights: {
        tight: number;
        normal: number;
        loose: number;
      };
    };
  }
}
