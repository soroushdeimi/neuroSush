import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Routes } from './app/routes';
import './design/tokens.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Routes />
  </StrictMode>
);
