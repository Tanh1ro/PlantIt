document.addEventListener("DOMContentLoaded", function () {
    let lastScrollTop = 0;
    let footer = document.querySelector("footer");

    // Hide footer initially
    footer.style.transform = "translateY(100%)";
    footer.style.transition = "transform 0.4s ease-in-out";

    window.addEventListener("scroll", function () {
        let scrollTop = window.pageYOffset || document.documentElement.scrollTop;

        if (scrollTop > lastScrollTop) {
            // User is scrolling down - Show Footer
            footer.style.transform = "translateY(0)";
        } else if (scrollTop < 100) {
            // User is near the top - Hide Footer
            footer.style.transform = "translateY(100%)";
        }

        lastScrollTop = scrollTop;
    });
});
