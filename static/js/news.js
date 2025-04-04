document.addEventListener('DOMContentLoaded', function() {
    // Create news container
    const newsContainer = document.createElement('div');
    newsContainer.className = 'news-container';
    
    // Create header
    const header = document.createElement('div');
    header.className = 'news-header';
    const title = document.createElement('h2');
    title.textContent = 'Latest Agricultural News';
    header.appendChild(title);
    newsContainer.appendChild(header);

    // Create news grid
    const newsGrid = document.createElement('div');
    newsGrid.className = 'news-grid';

    // Function to create news card
    function createNewsCard(article) {
        const card = document.createElement('div');
        card.className = 'news-card';
        
        const content = document.createElement('div');
        content.className = 'news-content';
        
        const title = document.createElement('h3');
        title.className = 'news-title';
        const titleLink = document.createElement('a');
        titleLink.href = article.url;
        titleLink.target = '_blank';
        titleLink.textContent = article.title;
        title.appendChild(titleLink);
        
        const summary = document.createElement('p');
        summary.className = 'news-summary';
        summary.textContent = article.summary;
        
        const readMore = document.createElement('a');
        readMore.className = 'news-link';
        readMore.href = article.url;
        readMore.target = '_blank';
        readMore.innerHTML = 'Read More <i class="fas fa-arrow-right"></i>';
        
        content.appendChild(title);
        content.appendChild(summary);
        content.appendChild(readMore);
        card.appendChild(content);
        
        return card;
    }

    // Function to handle no news available
    function createNoNewsMessage() {
        const noNews = document.createElement('div');
        noNews.className = 'no-news';
        const message = document.createElement('p');
        message.textContent = 'No news available at the moment. Please check back later.';
        noNews.appendChild(message);
        return noNews;
    }

    // Function to fetch and display news
    async function fetchAndDisplayNews() {
        try {
            const response = await fetch('/api/news'); // Replace with your actual API endpoint
            const articles = await response.json();
            
            if (articles && articles.length > 0) {
                articles.forEach(article => {
                    const card = createNewsCard(article);
                    newsGrid.appendChild(card);
                });
                newsContainer.appendChild(newsGrid);
            } else {
                const noNews = createNoNewsMessage();
                newsContainer.appendChild(noNews);
            }
        } catch (error) {
            console.error('Error fetching news:', error);
            const noNews = createNoNewsMessage();
            newsContainer.appendChild(noNews);
        }
    }

    // Add container to the page
    document.querySelector('.container').appendChild(newsContainer);

    // Fetch and display news
    fetchAndDisplayNews();

    // Add event listeners for animations
    const cards = document.querySelectorAll('.news-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = 'var(--box-shadow)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = 'none';
        });
    });
}); 