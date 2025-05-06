import React from 'react';
import styled from 'styled-components';
import Link from 'next/link';
import {
  Card,
  Button,
  Grid,
  SectionTitle,
  NeonText
} from '../components/ui/StyledComponents';

const HeroSection = styled.div`
  text-align: center;
  margin-bottom: ${({ theme }) => theme.spacing.xxl};
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: ${({ theme }) => theme.spacing.xxl} 0;
  animation: fadeIn 1s ease-out;
`;

const HeroTitle = styled.h1`
  font-size: ${({ theme }) => theme.typography.sizes.h1};
  margin-bottom: ${({ theme }) => theme.spacing.lg};
  color: ${({ theme }) => theme.colors.primary};
  font-weight: ${({ theme }) => theme.typography.fontWeights.bold};
  letter-spacing: 1px;
  animation: slideUp 0.8s ease-out;
  position: relative;

  &:after {
    content: '';
    position: absolute;
    bottom: -10px;
    left: 50%;
    transform: translateX(-50%);
    width: 80px;
    height: 3px;
    background-color: ${({ theme }) => theme.colors.primary};
  }
`;

const HeroSubtitle = styled.p`
  font-size: ${({ theme }) => theme.typography.sizes.xl};
  color: ${({ theme }) => theme.colors.textSecondary};
  margin-bottom: ${({ theme }) => theme.spacing.xl};
  max-width: 800px;
  line-height: ${({ theme }) => theme.typography.lineHeights.loose};
  animation: slideUp 0.8s ease-out 0.2s both;
  font-weight: ${({ theme }) => theme.typography.fontWeights.light};
`;

const FeatureCard = styled(Card)`
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  height: 100%;
  background-color: rgba(30, 30, 30, 0.4);
  transition: all ${({ theme }) => theme.transitions.medium};
  animation: fadeIn 0.8s ease-out;
  animation-fill-mode: both;
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.05);

  /* Staggered animation delays */
  &:nth-child(1) {
    animation-delay: 0.2s;
  }

  &:nth-child(2) {
    animation-delay: 0.4s;
  }

  &:nth-child(3) {
    animation-delay: 0.6s;
  }

  /* Subtle background gradient */
  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(
      circle at top right,
      rgba(255, 255, 255, 0.03) 0%,
      rgba(0, 0, 0, 0) 70%
    );
    z-index: 0;
  }

  /* Enhanced typography */
  h3 {
    margin: ${({ theme }) => theme.spacing.lg} 0;
    font-size: ${({ theme }) => theme.typography.sizes.xl};
    font-weight: ${({ theme }) => theme.typography.fontWeights.semibold};
    letter-spacing: 0.5px;
    position: relative;
    z-index: 1;
    transition: all ${({ theme }) => theme.transitions.medium};

    span {
      transition: all ${({ theme }) => theme.transitions.medium};
    }
  }

  &:hover h3 span {
    transform: translateY(-2px);
  }

  p {
    margin-bottom: ${({ theme }) => theme.spacing.xl};
    padding: 0 ${({ theme }) => theme.spacing.md};
    flex-grow: 1;
    color: ${({ theme }) => theme.colors.textSecondary};
    font-size: ${({ theme }) => theme.typography.sizes.lg};
    line-height: ${({ theme }) => theme.typography.lineHeights.loose};
    font-weight: ${({ theme }) => theme.typography.fontWeights.light};
    position: relative;
    z-index: 1;
  }

  /* Premium hover effect */
  &:hover {
    transform: translateY(-5px);
    background-color: rgba(40, 40, 40, 0.7);
    box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
    border-color: ${({ theme }) => theme.colors.primary}30;
  }

  /* Top border on hover */
  &:after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background-color: ${({ theme }) => theme.colors.primary};
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.4s cubic-bezier(0.23, 1, 0.32, 1);
  }

  &:hover:after {
    transform: scaleX(1);
  }

  /* Button animation */
  a {
    opacity: 0.9;
    transform: translateY(5px);
    transition: all ${({ theme }) => theme.transitions.medium};
  }

  &:hover a {
    opacity: 1;
    transform: translateY(0);
  }
`;

const FeatureIcon = styled.div`
  margin: ${({ theme }) => theme.spacing.xl} 0 ${({ theme }) => theme.spacing.md};
  display: flex;
  align-items: center;
  justify-content: center;
  width: 90px;
  height: 90px;
  border-radius: 50%;
  background-color: rgba(0, 0, 0, 0.2);
  position: relative;
  z-index: 1;
  transition: all ${({ theme }) => theme.transitions.medium};

  /* Subtle border */
  &:before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    border-radius: 50%;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: all ${({ theme }) => theme.transitions.medium};
  }

  svg {
    transition: all ${({ theme }) => theme.transitions.medium};
    transform: scale(0.9);
  }

  /* Enhanced hover effects */
  ${FeatureCard}:hover & {
    transform: translateY(-5px);
    background-color: rgba(0, 0, 0, 0.3);

    &:before {
      border-color: ${({ theme }) => theme.colors.primary}40;
    }

    svg {
      transform: scale(1);
    }
  }
`;

const PageContainer = styled.div`
  animation: fadeIn 0.5s ease-out;
  min-height: 100vh;
`;

const Home: React.FC = () => {
  return (
    <PageContainer>
      <HeroSection>
        <HeroTitle>TehqeeqCast</HeroTitle>
        <HeroSubtitle>
          A Stochastic Modeling Tool for Fake-News Propagation Prediction
        </HeroSubtitle>
        <Button
          variant="primary"
          as="a"
          href="/markov"
          style={{
            animation: 'slideUp 0.8s ease-out 0.4s both',
            marginTop: '20px'
          }}
        >
          Get Started
        </Button>
      </HeroSection>

      <SectionTitle>Our Models</SectionTitle>
      <Grid>
        <FeatureCard>
          <FeatureIcon>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="5" cy="12" r="2.5" fill="#FFC107" />
              <circle cx="12" cy="5" r="2.5" fill="#FFC107" />
              <circle cx="19" cy="12" r="2.5" fill="#FFC107" />
              <circle cx="12" cy="19" r="2.5" fill="#FFC107" />
              <line x1="5" y1="12" x2="12" y2="5" stroke="#FFC107" strokeWidth="1.5" />
              <line x1="12" y1="5" x2="19" y2="12" stroke="#FFC107" strokeWidth="1.5" />
              <line x1="19" y1="12" x2="12" y2="19" stroke="#FFC107" strokeWidth="1.5" />
              <line x1="12" y1="19" x2="5" y2="12" stroke="#FFC107" strokeWidth="1.5" />
            </svg>
          </FeatureIcon>
          <h3><NeonText color="primary">Markov Chain</NeonText></h3>
          <p>
            Model how misinformation hops across social media platforms with transition probabilities,
            steady-state analysis, and passage time calculations.
          </p>
          <Link href="/markov" passHref>
            <Button variant="primary" style={{ marginTop: '10px' }}>Explore Markov Chain</Button>
          </Link>
        </FeatureCard>

        <FeatureCard>
          <FeatureIcon>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="5" cy="5" r="1.8" fill="#FFC107" />
              <circle cx="5" cy="12" r="1.8" fill="#FFC107" />
              <circle cx="5" cy="19" r="1.8" fill="#FFC107" />
              <circle cx="19" cy="5" r="1.8" fill="#FFC107" />
              <circle cx="19" cy="12" r="1.8" fill="#FFC107" />
              <circle cx="19" cy="19" r="1.8" fill="#FFC107" />
              <line x1="5" y1="5" x2="19" y2="5" stroke="#FFC107" strokeWidth="1.5" strokeDasharray="2 2" />
              <line x1="5" y1="12" x2="19" y2="12" stroke="#FFC107" strokeWidth="1.5" strokeDasharray="2 2" />
              <line x1="5" y1="19" x2="19" y2="19" stroke="#FFC107" strokeWidth="1.5" strokeDasharray="2 2" />
              <line x1="5" y1="5" x2="5" y2="19" stroke="#FFC107" strokeWidth="1.5" />
              <line x1="19" y1="5" x2="19" y2="19" stroke="#FFC107" strokeWidth="1.5" />
            </svg>
          </FeatureIcon>
          <h3><NeonText color="secondary">Hidden Markov Model</NeonText></h3>
          <p>
            Infer each post's hidden credibility trajectory from observable engagement metrics
            using forward algorithms and Viterbi path analysis.
          </p>
          <Link href="/hmm" passHref>
            <Button variant="primary" style={{ marginTop: '10px' }}>Explore HMM</Button>
          </Link>
        </FeatureCard>

        <FeatureCard>
          <FeatureIcon>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="12" cy="12" r="9" stroke="#FFC107" strokeWidth="1.5" />
              <line x1="12" y1="3" x2="12" y2="12" stroke="#FFC107" strokeWidth="1.5" />
              <line x1="12" y1="12" x2="18" y2="12" stroke="#FFC107" strokeWidth="1.5" />
              <circle cx="12" cy="12" r="1.2" fill="#FFC107" />
            </svg>
          </FeatureIcon>
          <h3><NeonText color="accent">M/M/1 Queue</NeonText></h3>
          <p>
            Forecast moderation backlogs with queueing theory, analyzing arrival rates,
            service times, and system stability for fact-checking pipelines.
          </p>
          <Link href="/queue" passHref>
            <Button variant="primary" style={{ marginTop: '10px' }}>Explore M/M/1 Queue</Button>
          </Link>
        </FeatureCard>
      </Grid>
    </PageContainer>
  );
};

export default Home;
