import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Compose } from './modes/compose/Compose';
import './design/tokens.css';

createRoot(document.getElementById('root')!).render(<StrictMode><Compose /></StrictMode>);
