import { useState, useEffect, lazy, Suspense } from 'react';
import { Compose } from '../modes/compose/Compose';

const Workspace = lazy(() => import('../modes/workspace/index'));
const Inspect = lazy(() => import('../modes/inspect/index'));
const Tour = lazy(() => import('../modes/tour/index'));
const Forms = lazy(() => import('../modes/forms/index'));

export function Routes() {
  const [hash, setHash] = useState(window.location.hash || '#/compose');

  useEffect(() => {
    const onHashChange = () => {
      setHash(window.location.hash || '#/compose');
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const current = hash.startsWith('#/') ? hash : '#/compose';

  return (
    <Suspense fallback={<div className="status">Loading mode…</div>}>
      {current === '#/compose' && <Compose />}
      {current === '#/workspace' && <Workspace />}
      {current === '#/inspect' && <Inspect />}
      {current === '#/tour' && <Tour />}
      {current === '#/forms' && <Forms />}
    </Suspense>
  );
}
