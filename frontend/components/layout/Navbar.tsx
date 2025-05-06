import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import Link from 'next/link';
import { useRouter } from 'next/router';

const NavbarContainer = styled.nav`
  background-color: ${({ theme }) => theme.colors.background};
  padding: ${({ theme }) => theme.spacing.md} ${({ theme }) => theme.spacing.xl};
  box-shadow: ${({ theme }) => theme.shadows.medium};
  position: sticky;
  top: 0;
  z-index: 100;
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  animation: fadeIn 0.5s ease-in-out;
`;

const Logo = styled.div`
  font-size: ${({ theme }) => theme.typography.sizes.h3};
  font-weight: ${({ theme }) => theme.typography.fontWeights.bold};
  color: ${({ theme }) => theme.colors.primary};
  letter-spacing: 1px;
  position: relative;

  &:after {
    content: '';
    position: absolute;
    bottom: -4px;
    left: 0;
    width: 30%;
    height: 2px;
    background-color: ${({ theme }) => theme.colors.primary};
    transition: width ${({ theme }) => theme.transitions.medium};
  }

  &:hover:after {
    width: 100%;
  }
`;

const NavLinks = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.xl};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    display: none;
  }
`;

const NavLink = styled.a<{ active?: boolean }>`
  color: ${({ theme, active }) => active ? theme.colors.primary : theme.colors.textPrimary};
  text-decoration: none;
  position: relative;
  padding: ${({ theme }) => theme.spacing.sm} ${({ theme }) => theme.spacing.md};
  font-weight: ${({ theme, active }) => active ? theme.typography.fontWeights.semibold : theme.typography.fontWeights.regular};
  font-size: ${({ theme }) => theme.typography.sizes.lg};
  letter-spacing: 0.5px;
  transition: all ${({ theme }) => theme.transitions.medium};

  &:after {
    content: '';
    position: absolute;
    width: ${({ active }) => active ? '100%' : '0'};
    height: 2px;
    bottom: -2px;
    left: 0;
    background-color: ${({ theme }) => theme.colors.primary};
    transition: width ${({ theme }) => theme.transitions.medium};
  }

  &:hover {
    color: ${({ theme }) => theme.colors.primary};
    transform: translateY(-2px);
  }

  &:hover:after {
    width: 100%;
  }
`;

const MobileMenuButton = styled.button`
  background: none;
  border: none;
  color: ${({ theme }) => theme.colors.primary};
  font-size: ${({ theme }) => theme.typography.sizes.h4};
  cursor: pointer;
  display: none;
  transition: transform ${({ theme }) => theme.transitions.fast};

  &:hover {
    transform: scale(1.1);
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    display: block;
  }
`;

const MobileMenu = styled.div<{ isOpen: boolean }>`
  position: fixed;
  top: 70px;
  left: 0;
  right: 0;
  background-color: ${({ theme }) => theme.colors.background};
  backdrop-filter: blur(10px);
  padding: ${({ theme }) => theme.spacing.lg};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing.md};
  transform: translateY(${({ isOpen }) => (isOpen ? '0' : '-100%')});
  opacity: ${({ isOpen }) => (isOpen ? '1' : '0')};
  transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.3s ease;
  box-shadow: ${({ theme }) => theme.shadows.medium};
  z-index: 99;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
`;

const MobileNavLink = styled(NavLink)`
  padding: ${({ theme }) => theme.spacing.md};
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  font-size: ${({ theme }) => theme.typography.sizes.xl};
  text-align: center;

  &:last-child {
    border-bottom: none;
  }

  &:hover {
    background-color: rgba(255, 255, 255, 0.03);
    transform: translateY(0);
  }
`;

const Navbar: React.FC = () => {
  const router = useRouter();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Close mobile menu when route changes
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [router.pathname]);

  const isActive = (path: string) => router.pathname === path;

  return (
    <>
      <NavbarContainer>
        <Logo>TehqeeqCast</Logo>

        <NavLinks>
          <Link href="/" passHref legacyBehavior>
            <NavLink active={isActive('/')}>Home</NavLink>
          </Link>
          <Link href="/markov" passHref legacyBehavior>
            <NavLink active={isActive('/markov')}>Markov Chain</NavLink>
          </Link>
          <Link href="/hmm" passHref legacyBehavior>
            <NavLink active={isActive('/hmm')}>Hidden Markov Model</NavLink>
          </Link>
          <Link href="/queue" passHref legacyBehavior>
            <NavLink active={isActive('/queue')}>M/M/1 Queue</NavLink>
          </Link>
        </NavLinks>

        <MobileMenuButton onClick={() => setMobileMenuOpen(!mobileMenuOpen)}>
          {mobileMenuOpen ? '✕' : '☰'}
        </MobileMenuButton>
      </NavbarContainer>

      <MobileMenu isOpen={mobileMenuOpen}>
        <Link href="/" passHref legacyBehavior>
          <MobileNavLink active={isActive('/')}>Home</MobileNavLink>
        </Link>
        <Link href="/markov" passHref legacyBehavior>
          <MobileNavLink active={isActive('/markov')}>Markov Chain</MobileNavLink>
        </Link>
        <Link href="/hmm" passHref legacyBehavior>
          <MobileNavLink active={isActive('/hmm')}>Hidden Markov Model</MobileNavLink>
        </Link>
        <Link href="/queue" passHref legacyBehavior>
          <MobileNavLink active={isActive('/queue')}>M/M/1 Queue</MobileNavLink>
        </Link>
      </MobileMenu>
    </>
  );
};

export default Navbar;
