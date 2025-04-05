document.addEventListener('DOMContentLoaded', function() {
    if (typeof tailwind !== 'undefined') {
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        primary: {
                            600: '#0284c7',
                            700: '#0369a1'
                        }
                    }
                }
            }
        };
    }
});