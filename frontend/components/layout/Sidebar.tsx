import React from 'react';
import styled from 'styled-components';
import Link from 'next/link';
import { useRouter } from 'next/router';

const SidebarContainer = styled.aside`
  background-color: ${({ theme }) => theme.colors.surface};
  width: 250px;
  height: 100vh;
  position: fixed;
  top: 0;
  left: 0;
  padding-top: 80px; /* Space for navbar */
  box-shadow: ${({ theme }) => theme.shadows.medium};
  
  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    width: 200px;
  }
  
  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    display: none;
  }
`;

const SidebarNav = styled.nav`
  display: flex;
  flex-direction: column;
  padding: ${({ theme }) => theme.spacing.md};
`;

const SidebarLink = styled.a<{ active?: boolean }>`
  color: ${({ theme, active }) => active ? theme.colors.primary : theme.colors.textPrimary};
  text-decoration: none;
  padding: ${({ theme }) => theme.spacing.md};
  border-radius: ${({ theme }) => theme.borderRadius.small};
  margin-bottom: ${({ theme }) => theme.spacing.sm};
  background-color: ${({ theme, active }) => active ? `${theme.colors.primary}20` : 'transparent'};
  transition: background-color ${({ theme }) => theme.transitions.fast};
  
  &:hover {
    background-color: ${({ theme, active }) => active ? `${theme.colors.primary}20` : `${theme.colors.surfaceLight}`};
  }
`;

const SidebarSection = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing.lg};
`;

const SidebarSectionTitle = styled.h3`
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 1px;
  padding: ${({ theme }) => `${theme.spacing.md} ${theme.spacing.md} ${theme.spacing.xs}`};
  margin: 0;
`;

const Sidebar: React.FC = () => {
  const router = useRouter();
  
  const isActive = (path: string) => router.pathname === path;
  
  return (
    <SidebarContainer>
      <SidebarNav>
        <SidebarSection>
          <SidebarSectionTitle>Models</SidebarSectionTitle>
          <Link href="/markov" passHref legacyBehavior>
            <SidebarLink active={isActive('/markov')}>Markov Chain</SidebarLink>
          </Link>
          <Link href="/hmm" passHref legacyBehavior>
            <SidebarLink active={isActive('/hmm')}>Hidden Markov Model</SidebarLink>
          </Link>
          <Link href="/queue" passHref legacyBehavior>
            <SidebarLink active={isActive('/queue')}>M/M/1 Queue</SidebarLink>
          </Link>
        </SidebarSection>
        
        <SidebarSection>
          <SidebarSectionTitle>Resources</SidebarSectionTitle>
          <Link href="/docs" passHref legacyBehavior>
            <SidebarLink active={isActive('/docs')}>Documentation</SidebarLink>
          </Link>
          <Link href="/about" passHref legacyBehavior>
            <SidebarLink active={isActive('/about')}>About</SidebarLink>
          </Link>
        </SidebarSection>
      </SidebarNav>
    </SidebarContainer>
  );
};

export default Sidebar;
