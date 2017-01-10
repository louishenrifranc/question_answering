(function() {
  
  "use strict";

  angular.module("qa_app", [])

    .controller("qa_controller", ["$scope", "$log", "$http", 
                                  function($scope, $log, $http, $timeout) {
                                    $scope.ask = function() {
                                      $http.post("/ask", {"context": $scope.context,
                                                         "question": $scope.question}).
                                        success(function(results) {
                                          $log.log(results)
                                          $scope.answer = results.answer
                                          $scope.attention_context = results.context
                                        }).
                                        error(function(error) {
                                          $log.log(error);
                                        });
                                    };
                                  }
                                 ]);
}());
