import styled from 'styled-components';

export const Card = styled.div`
  background-color: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.borderRadius.medium};
  padding: ${({ theme }) => theme.spacing.xl};
  box-shadow: ${({ theme }) => theme.shadows.medium};
  margin-bottom: ${({ theme }) => theme.spacing.lg};
  border: 1px solid rgba(255, 255, 255, 0.05);
  transition: all ${({ theme }) => theme.transitions.medium};
  overflow: hidden;
  position: relative;
  backdrop-filter: blur(5px);

  &:hover {
    transform: translateY(-2px);
    box-shadow: ${({ theme }) => theme.shadows.large};
    border-color: ${({ theme }) => theme.colors.primary}30;
  }

  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background-color: ${({ theme }) => theme.colors.primary};
    opacity: 0;
    transition: opacity ${({ theme }) => theme.transitions.medium};
  }

  &:hover:before {
    opacity: 1;
  }

  &:after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at top right,
      rgba(255, 255, 255, 0.03) 0%,
      rgba(255, 255, 255, 0) 70%);
    pointer-events: none;
  }
`;

export const GlassCard = styled(Card)`
  background-color: rgba(30, 30, 30, 0.7);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
`;

export const Button = styled.button<{ variant?: 'primary' | 'secondary' | 'accent' }>`
  background-color: ${({ theme, variant = 'primary' }) => theme.colors[variant]};
  color: ${({ theme }) => theme.colors.background};
  border: none;
  border-radius: ${({ theme }) => theme.borderRadius.small};
  padding: ${({ theme }) => `${theme.spacing.md} ${theme.spacing.xl}`};
  font-weight: ${({ theme }) => theme.typography.fontWeights.medium};
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.medium};
  text-transform: uppercase;
  letter-spacing: 2px;
  font-size: ${({ theme }) => theme.typography.sizes.md};
  position: relative;
  overflow: hidden;

  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(255, 255, 255, 0.1);
    transform: translateX(-100%);
    transition: transform ${({ theme }) => theme.transitions.medium};
  }

  &:hover {
    transform: translateY(-3px);
    background-color: ${({ theme, variant = 'primary' }) => theme.colors[variant]};
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);

    &:before {
      transform: translateX(0);
    }
  }

  &:active {
    transform: translateY(1px);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }
`;

export const Input = styled.input`
  background-color: ${({ theme }) => theme.colors.surfaceLight};
  color: ${({ theme }) => theme.colors.textPrimary};
  border: 1px solid ${({ theme }) => theme.colors.surfaceLight};
  border-radius: ${({ theme }) => theme.borderRadius.small};
  padding: ${({ theme }) => theme.spacing.sm};
  width: 100%;
  transition: border-color ${({ theme }) => theme.transitions.fast};

  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.colors.primary};
    box-shadow: 0 0 0 2px ${({ theme }) => theme.colors.primary}40;
  }
`;

export const Select = styled.select`
  background-color: ${({ theme }) => theme.colors.surfaceLight};
  color: ${({ theme }) => theme.colors.textPrimary};
  border: 1px solid ${({ theme }) => theme.colors.surfaceLight};
  border-radius: ${({ theme }) => theme.borderRadius.small};
  padding: ${({ theme }) => theme.spacing.sm};
  width: 100%;
  transition: border-color ${({ theme }) => theme.transitions.fast};

  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.colors.primary};
    box-shadow: 0 0 0 2px ${({ theme }) => theme.colors.primary}40;
  }
`;

export const TextArea = styled.textarea`
  background-color: ${({ theme }) => theme.colors.surfaceLight};
  color: ${({ theme }) => theme.colors.textPrimary};
  border: 1px solid ${({ theme }) => theme.colors.surfaceLight};
  border-radius: ${({ theme }) => theme.borderRadius.small};
  padding: ${({ theme }) => theme.spacing.sm};
  width: 100%;
  min-height: 100px;
  transition: border-color ${({ theme }) => theme.transitions.fast};

  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.colors.primary};
    box-shadow: 0 0 0 2px ${({ theme }) => theme.colors.primary}40;
  }
`;

export const FormGroup = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing.md};
`;

export const Label = styled.label`
  display: block;
  margin-bottom: ${({ theme }) => theme.spacing.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
`;

export const Grid = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: ${({ theme }) => theme.spacing.lg};

  @media (max-width: ${({ theme }) => theme.breakpoints.xl}) {
    grid-template-columns: repeat(2, 1fr);
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: 1fr;
  }
`;

export const Flex = styled.div<{ direction?: 'row' | 'column', justify?: string, align?: string }>`
  display: flex;
  flex-direction: ${({ direction = 'row' }) => direction};
  justify-content: ${({ justify = 'flex-start' }) => justify};
  align-items: ${({ align = 'stretch' }) => align};
`;

export const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: ${({ theme }) => theme.spacing.lg};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    padding: ${({ theme }) => theme.spacing.md};
  }
`;

export const PageTitle = styled.h1`
  margin-bottom: ${({ theme }) => theme.spacing.xl};
  font-size: 2.5rem;
  font-weight: bold;
  background: linear-gradient(to right,
    ${({ theme }) => theme.colors.primary},
    ${({ theme }) => theme.colors.accent}
  );
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-align: center;
`;

export const SectionTitle = styled.h2`
  margin-bottom: ${({ theme }) => theme.spacing.xxl};
  font-size: ${({ theme }) => theme.typography.sizes.h2};
  color: ${({ theme }) => theme.colors.textPrimary};
  text-align: center;
  position: relative;
  padding-bottom: ${({ theme }) => theme.spacing.md};
  font-weight: ${({ theme }) => theme.typography.fontWeights.bold};
  letter-spacing: 1px;
  animation: fadeIn 0.8s ease-out;

  &:after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 80px;
    height: 3px;
    background-color: ${({ theme }) => theme.colors.primary};
    animation: slideUp 0.5s ease-out 0.3s both;
  }
`;

export const ErrorMessage = styled.p`
  color: ${({ theme }) => theme.colors.error};
  margin-top: ${({ theme }) => theme.spacing.xs};
  font-size: 0.9rem;
`;

export const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  margin-bottom: ${({ theme }) => theme.spacing.lg};
`;

export const TableHead = styled.thead`
  background-color: ${({ theme }) => theme.colors.surfaceLight};
`;

export const TableRow = styled.tr`
  border-bottom: 1px solid ${({ theme }) => theme.colors.surfaceLight};

  &:hover {
    background-color: rgba(255, 255, 255, 0.05);
  }
`;

export const TableCell = styled.td`
  padding: ${({ theme }) => theme.spacing.sm};
`;

export const TableHeaderCell = styled.th`
  padding: ${({ theme }) => theme.spacing.sm};
  text-align: left;
  font-weight: bold;
`;

export const Badge = styled.span<{ variant?: 'primary' | 'secondary' | 'accent' | 'error' }>`
  background-color: ${({ theme, variant = 'primary' }) => theme.colors[variant]};
  color: ${({ theme }) => theme.colors.background};
  padding: ${({ theme }) => `${theme.spacing.xs} ${theme.spacing.sm}`};
  border-radius: ${({ theme }) => theme.borderRadius.small};
  font-size: 0.8rem;
  font-weight: bold;
`;

export const Divider = styled.hr`
  border: none;
  border-top: 1px solid ${({ theme }) => theme.colors.surfaceLight};
  margin: ${({ theme }) => theme.spacing.lg} 0;
`;

export const ImageContainer = styled.div`
  width: 100%;
  margin: ${({ theme }) => theme.spacing.md} 0;
  border-radius: ${({ theme }) => theme.borderRadius.medium};
  overflow: hidden;

  img {
    width: 100%;
    height: auto;
    display: block;
  }
`;

export const NeonText = styled.span<{ color?: 'primary' | 'secondary' | 'accent' }>`
  color: ${({ theme }) => theme.colors.primary};
  font-weight: ${({ theme }) => theme.typography.fontWeights.semibold};
  letter-spacing: 1px;
  position: relative;
  display: inline-block;
  transition: all ${({ theme }) => theme.transitions.medium};
`;

export const Spinner = styled.div`
  border: 4px solid rgba(255, 255, 255, 0.1);
  border-radius: 50%;
  border-top: 4px solid ${({ theme }) => theme.colors.primary};
  width: 30px;
  height: 30px;
  animation: spin 1s linear infinite;
  margin: ${({ theme }) => theme.spacing.md} auto;

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

export const FileUpload = styled.div`
  border: 2px dashed ${({ theme }) => theme.colors.surfaceLight};
  border-radius: ${({ theme }) => theme.borderRadius.medium};
  padding: ${({ theme }) => theme.spacing.lg};
  text-align: center;
  cursor: pointer;
  transition: border-color ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme }) => theme.colors.primary};
  }

  input {
    display: none;
  }
`;
