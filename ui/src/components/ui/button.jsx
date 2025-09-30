export function Button({ variant='default', size='md', className='', ...props }) {
  const base = 'inline-flex items-center justify-center rounded-xl border text-sm font-medium transition px-3 py-2';
  const styles = {
    default: 'bg-black text-white hover:opacity-90',
    secondary: 'bg-gray-100 text-gray-900 border-gray-200 hover:bg-gray-200'
  }[variant] || '';
  return <button className={`${base} ${styles} ${className}`} {...props} />;
}
