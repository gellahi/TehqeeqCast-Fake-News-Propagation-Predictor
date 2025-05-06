import React from 'react';
import styled, { ThemeProvider } from 'styled-components';
import { theme, GlobalStyle } from '../../styles/theme';
import Navbar from './Navbar';

const LayoutContainer = styled.div`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
`;

const MainContent = styled.main`
  flex: 1;
  padding: ${({ theme }) => theme.spacing.xl};
  width: 100%;

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    padding: ${({ theme }) => theme.spacing.lg};
  }
`;

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle theme={theme} />
      <LayoutContainer>
        <Navbar />
        <MainContent>{children}</MainContent>
      </LayoutContainer>
    </ThemeProvider>
  );
};

export default Layout;
