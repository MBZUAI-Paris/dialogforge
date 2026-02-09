document$.subscribe(function () {
  var path = window.location.pathname || "";
  var isApiOverview =
    /\/reference\/?$/.test(path) || /\/reference\/index\.html$/.test(path);

  if (!isApiOverview) {
    return;
  }

  var labels = document.querySelectorAll(".md-nav__link[for]");
  labels.forEach(function (label) {
    var sectionName = label.textContent.trim();
    if (sectionName !== "dlgforge") {
      return;
    }

    var toggleId = label.getAttribute("for");
    var toggle = toggleId ? document.getElementById(toggleId) : null;
    if (toggle) {
      toggle.checked = false;
    }

    var nestedNav = label.nextElementSibling;
    if (nestedNav && nestedNav.classList.contains("md-nav")) {
      nestedNav.setAttribute("aria-expanded", "false");
    }
  });
});
