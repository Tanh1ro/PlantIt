document.addEventListener("DOMContentLoaded", function () {
    let lastScrollTop = 0;
    const navbar = document.getElementById("navbar");

    window.addEventListener("scroll", function () {
        let scrollTop = window.pageYOffset || document.documentElement.scrollTop;

        if (scrollTop > lastScrollTop) {
            // Scrolling down - hide navbar
            navbar.style.top = "-80px";  // Adjust based on navbar height
        } else {
            // Scrolling up - show navbar
            navbar.style.top = "0";
        }

        lastScrollTop = scrollTop;
    });
});
