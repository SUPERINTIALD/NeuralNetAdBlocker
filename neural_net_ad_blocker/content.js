(async function() {
  console.log('Waiting for dynamic content...');
  await new Promise(resolve => setTimeout(resolve, 3000)); // wait 3 seconds

  const html = document.documentElement.outerHTML;

  try {
    const response = await fetch('http://127.0.0.1:5555/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ html: html })
    });

    if (response.ok) {
      const modifiedHtml = await response.text();
      document.open();
      document.write(modifiedHtml);
      document.close();
    } else {
      console.error('Server error:', response.status);
    }
  } catch (error) {
    console.error('Fetch failed:', error);
  }
})();
