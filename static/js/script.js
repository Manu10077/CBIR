document.addEventListener("DOMContentLoaded", () => {
  // Animate floating dataset images
  const floats = document.querySelectorAll(".float-item");
  floats.forEach((item) => {
    const x1 = Math.random() * 100 + "%";
    const y1 = Math.random() * 100 + "%";
    const x2 = Math.random() * 100 + "%";
    const y2 = Math.random() * 100 + "%";
    const x3 = Math.random() * 100 + "%";
    const y3 = Math.random() * 100 + "%";
    const duration = 50 + Math.random() * 80;
    item.style.setProperty("--x1", x1);
    item.style.setProperty("--y1", y1);
    item.style.setProperty("--x2", x2);
    item.style.setProperty("--y2", y2);
    item.style.setProperty("--x3", x3);
    item.style.setProperty("--y3", y3);
    item.style.animationDuration = duration + "s";
  });
});
// Optional parallax motion effect for background rows
document.addEventListener("mousemove", (e) => {
  const rows = document.querySelectorAll(".image-row");
  rows.forEach((row, i) => {
    const speed = (i + 1) * 0.05;
    const x = (window.innerWidth / 2 - e.pageX) * speed;
    row.style.transform = `translateX(${x}px)`;
  });
});
