import React from 'react';
export const Input = React.forwardRef(function Input({ className='', ...props }, ref) {
  return <input ref={ref} className={`w-full rounded-lg border border-gray-300 px-3 py-2 outline-none focus:ring-2 focus:ring-black/20 ${className}`} {...props} />;
});
